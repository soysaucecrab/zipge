"""Prune multilingual-e5-small's vocabulary to the site's languages.

Keeps Hangul/short-Latin/digit/punct pieces + every piece the corpus uses
(data/bench/keep_ids.npy), slices the embedding matrix, remaps the fast
tokenizer, exports ONNX, and int8-quantizes. Output layout matches what
transformers.js expects (config.json / tokenizer.json / onnx/model_quantized.onnx).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
KEEP = ROOT / "data" / "bench" / "keep_ids.npy"
WORK = ROOT / "data" / "pruned_model"
HF_DIR = WORK / "hf"
OUT = WORK / "e5-small-ko"

MODEL_ID = "intfloat/multilingual-e5-small"


def remap_ids(obj, old2new):
    """Recursively rewrite integer token ids in tokenizer post-processor blobs."""
    if isinstance(obj, dict):
        return {k: (old2new[v] if k in ("id",) and isinstance(v, int) else remap_ids(v, old2new))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_ids(x, old2new) for x in obj]
    return obj


def main():
    keep = [int(i) for i in np.load(KEEP)]
    old2new = {o: n for n, o in enumerate(keep)}

    print(f"pruning to {len(keep)} pieces")
    model = AutoModel.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    with torch.no_grad():
        emb = model.embeddings.word_embeddings
        new_emb = torch.nn.Embedding(len(keep), emb.embedding_dim, padding_idx=old2new.get(1))
        new_emb.weight.copy_(emb.weight[torch.tensor(keep)])
        model.embeddings.word_embeddings = new_emb
    model.config.vocab_size = len(keep)

    HF_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(HF_DIR)
    tok.save_pretrained(HF_DIR)

    # --- fast tokenizer surgery ---
    tj_path = HF_DIR / "tokenizer.json"
    tj = json.loads(tj_path.read_text(encoding="utf-8"))
    vocab_list = tj["model"]["vocab"]  # [[piece, score], ...] indexed by id
    tj["model"]["vocab"] = [vocab_list[i] for i in keep]
    if "unk_id" in tj["model"]:
        tj["model"]["unk_id"] = old2new[tj["model"]["unk_id"]]
    for t in tj.get("added_tokens", []):
        t["id"] = old2new[t["id"]]
    if tj.get("post_processor"):
        tj["post_processor"] = remap_ids(tj["post_processor"], old2new)
    tj_path.write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")

    # sentencepiece binary would resurrect old ids; the fast tokenizer is the truth
    for junk in ["sentencepiece.bpe.model"]:
        p = HF_DIR / junk
        if p.exists():
            p.unlink()

    # --- ONNX export + int8 quantize ---
    onnx_dir = WORK / "onnx_export"
    subprocess.run([sys.executable, "-m", "optimum.exporters.onnx", "--model", str(HF_DIR),
                    "--task", "feature-extraction", str(onnx_dir)], check=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "onnx").mkdir(parents=True)
    quantize_dynamic(onnx_dir / "model.onnx", OUT / "onnx" / "model_quantized.onnx",
                     weight_type=QuantType.QInt8)
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (HF_DIR / f).exists():
            shutil.copy(HF_DIR / f, OUT / f)

    size = (OUT / "onnx" / "model_quantized.onnx").stat().st_size / 1e6
    print(f"done: {OUT} (model_quantized.onnx {size:.0f}MB)")


if __name__ == "__main__":
    main()
