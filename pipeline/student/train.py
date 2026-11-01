"""Student encoder — step 3/4: distil the teacher into a model-free query encoder and export it.

    student(query) = normalize(W · Σ_slots w_slot·v_slot + b)
      core slot      w = softplus(θ_t)·exp(fbw·fb_slot), v = core term vector (vocab_input.npz)
      lexicon slot   w = softplus(extw),                  v = extension-lexicon vector (lexicon.npz)
      θ_t per-term weight (init: IDF), fbw: fallback (substring) slots, extw: one shared weight for
      out-of-corpus words (build_lexicon.py; used only when its basis matches vocab_input's)
    loss = KL(softmax(student·D/τ) ‖ softmax(teacher·D/τ)) + λ·(1 − cos(student, teacher))
           + aux·CE(student·D/τ, source doc)

(W, b) start from the ridge-regression solution, then Adam with early stopping on a 5%
hold-out; N seeds are averaged at query time (ensemble members).

Outputs: <work>/checkpoints/seed{N}.pt and the deployed assets in --out (common.write_assets).
`--export-only` rebuilds the assets from existing checkpoints (no training data needed).

Run: python pipeline/student/train.py [--work data/student --index data/index]
"""
import argparse
import hashlib
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from common import (StudentRuntime, build_slots, load_json, make_idf, metrics, ranks,  # noqa: E402
                    write_assets, write_json)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAM_KEYS = ("theta", "fbw", "extw", "lin.weight", "lin.bias")


class Student(nn.Module):
    """V = core vocabulary rows followed by the extension-lexicon rows the data uses
    (slot idx >= n_core is a lexicon slot)."""

    def __init__(self, V, idf_init, out_dim, n_core, ext_init):
        super().__init__()
        self.register_buffer("V", torch.tensor(V, dtype=torch.float32))
        self.n_core = n_core
        self.theta = nn.Parameter(torch.log(torch.expm1(torch.tensor(idf_init, dtype=torch.float32))))
        self.fbw = nn.Parameter(torch.tensor(-0.5))
        self.extw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(ext_init)))))
        self.lin = nn.Linear(V.shape[1], out_dim)

    def pool(self, B):
        idx = B["idx"]
        is_ext = idx >= self.n_core
        core_w = F.softplus(self.theta[torch.where(is_ext, 0, idx)]) * torch.exp(self.fbw * B["fb"])
        w = torch.where(is_ext, F.softplus(self.extw), core_w) * B["mask"]
        return (w.unsqueeze(-1) * self.V[idx]).sum(1)

    def forward(self, B):
        return F.normalize(self.lin(self.pool(B)), dim=-1)


def to_batch(idx, mask, fb):
    return {"idx": torch.tensor(idx, device=DEV), "mask": torch.tensor(mask, device=DEV).float(),
            "fb": torch.tensor(fb, device=DEV)}


def load_lexicon(work, basis_sha1):
    """Extension lexicon for training, only if it lives in the same input basis as vocab_input."""
    p = Path(work) / "lexicon" / "lexicon.npz"
    if not basis_sha1 or not p.exists():
        return None, None
    lx = np.load(p)
    if str(lx["basis_sha1"]) != basis_sha1:
        print("lexicon basis differs from vocab_input — rebuild the lexicon; training without it")
        return None, None
    terms = [str(t) for t in lx["terms"]]
    return {t: i for i, t in enumerate(terms)}, lx["q"].astype(np.float32) * lx["scale"][:, None]


def compact_ext(idx, ext, n_core, lex_vec):
    """Remap lexicon slots onto rows n_core.. of a compact table holding only the lexicon rows used."""
    used = np.unique(idx[ext])
    remap = np.full(len(lex_vec), -1, np.int64)
    remap[used] = np.arange(len(used)) + n_core
    out = idx.copy()
    out[ext] = remap[idx[ext]]
    return out, lex_vec[used]


def take(B, sel):
    return {k: v[sel] for k, v in B.items()}


@torch.no_grad()
def ridge_init(model, B, T):
    n = B["idx"].shape[0]
    P = torch.cat([model.pool(take(B, torch.arange(i, min(i + 4096, n), device=DEV))) for i in range(0, n, 4096)])
    P1 = torch.cat([P, torch.ones(n, 1, device=DEV)], 1)
    G = P1.T @ P1
    lam = 1e-2 * torch.trace(G) / G.shape[0]
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV), P1.T @ T)
    model.lin.weight.copy_(W[:-1].T)
    model.lin.bias.copy_(W[-1])


def train_one(model, B, T, src, D, cfg, seed, colmask=None):
    torch.manual_seed(seed)
    random.seed(seed)
    ridge_init(model, B, T)
    n = B["idx"].shape[0]
    perm = torch.randperm(n, device=DEV)
    nval = max(500, n // 20)
    vsel, tsel = perm[:nval], perm[nval:]
    opt = torch.optim.Adam([
        {"params": [model.theta, model.fbw, model.extw], "lr": 2e-2},
        {"params": model.lin.parameters(), "lr": 2e-3},
    ])
    tau = cfg["tau"]

    def logits(q):
        L = (q @ D.T) / tau
        return L if colmask is None else L.masked_fill(colmask[None, :], -1e9)

    def loss_fn(sel):
        q, t = model(take(B, sel)), T[sel]
        loss = F.kl_div(F.log_softmax(logits(q), -1), F.log_softmax(logits(t), -1), log_target=True,
                        reduction="batchmean")
        if cfg["lam"]:
            loss = loss + cfg["lam"] * (1 - (q * t).sum(-1)).mean()
        if cfg["aux"]:
            s = src[sel]
            ok = s >= 0
            if ok.any():
                loss = loss + cfg["aux"] * F.cross_entropy(logits(q[ok]), s[ok])
        return loss

    best, state, bad, ep = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = tsel[torch.randperm(len(tsel), device=DEV)]
        for i in range(0, len(order), cfg["batch"]):
            opt.zero_grad()
            loss_fn(order[i:i + cfg["batch"]]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(vsel[i:i + 2048]).item() for i in range(0, len(vsel), 2048))
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(state)
    return model.eval(), best, ep + 1


def sha1(items):
    return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "data" / "student"))
    ap.add_argument("--index", default=str(ROOT / "data" / "index"),
                    help="dir with doc_meta.json (row order of the doc vectors)")
    ap.add_argument("--out", help="asset output dir (default: --index)")
    ap.add_argument("--teacher", default="BAAI/bge-m3", help="recorded in the asset meta")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-fallback", action="store_true", help="ablation: drop substring fallback slots")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="experiment: hide this fraction of docs from training (new-article generalisation)")
    ap.add_argument("--export-only", action="store_true", help="rebuild assets from the checkpoint dir")
    ap.add_argument("--ckpt-dir", help="default <work>/checkpoints (holdout runs: <work>/checkpoints_holdout)")
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out or a.index)
    # holdout runs are experiments: never overwrite the production checkpoints
    ck_dir = Path(a.ckpt_dir) if a.ckpt_dir else work / ("checkpoints_holdout" if a.holdout_frac > 0 else "checkpoints")
    ck_dir.mkdir(parents=True, exist_ok=True)
    if a.holdout_frac > 0 and out.resolve() == Path(a.index).resolve():
        sys.exit("--holdout-frac is an experiment: pass a separate --out so the shipped assets stay intact")
    cfg = {k: getattr(a, k) for k in ("tau", "lam", "aux", "epochs", "batch", "patience", "holdout_frac")}
    cfg["fallback"] = not a.no_fallback

    vocab = load_json(work / "vocab.json")
    terms = vocab["terms"]
    idf = make_idf(vocab)
    vi = {t: i for i, t in enumerate(terms)}
    vin = np.load(work / "vocab_input.npz")
    Vq, Vs = vin["q"], vin["scale"]
    V = Vq.astype(np.float32) * Vs[:, None]
    basis_sha1 = str(vin["basis_sha1"]) if "basis_sha1" in vin.files else ""
    lex, lex_vec = load_lexicon(work, basis_sha1)
    if lex is not None:
        print(f"extension lexicon: {len(lex)} words (basis {basis_sha1[:12]})")
    D_np = np.load(work / "teacher_docs.npy")
    D = torch.tensor(D_np, device=DEV)
    doc_ids = [str(m["id"]) for m in load_json(Path(a.index) / "doc_meta.json")]
    if len(doc_ids) != D.shape[0]:
        sys.exit(f"doc_meta rows {len(doc_ids)} != teacher_docs rows {D.shape[0]} — rerun embed_teacher.py")
    fingerprint = {"terms_sha1": sha1(terms), "doc_ids_sha1": sha1(doc_ids)}

    states = []
    if a.export_only:
        for p in sorted(ck_dir.glob("seed*.pt")):
            ck = torch.load(p, map_location="cpu")
            if ck["fingerprint"] != fingerprint:
                sys.exit(f"{p.name} was trained on a different vocab/doc set — retrain")
            if ck.get("basis_sha1", "") != basis_sha1:
                sys.exit(f"{p.name} was trained in a different input basis (lexicon rebuilt?) — retrain")
            states.append(ck["state_dict"])
            cfg = ck["config"]
        if not states:
            sys.exit(f"no checkpoints in {ck_dir}")
        print(f"export-only: {len(states)} checkpoints")
    else:
        train = load_json(work / "train_queries.json")
        T_all = torch.tensor(np.load(work / "teacher_q_train.npy").astype(np.float32), device=DEV)
        idx, mask, fb, ext, _ = build_slots(train, vi, use_fallback=cfg["fallback"], lex=lex)
        if lex is not None:
            idx, ext_rows = compact_ext(idx, ext & mask, len(terms), lex_vec)
            V_train = np.concatenate([V, ext_rows])
            print(f"lexicon slots in training: {int((ext & mask).sum())} ({len(ext_rows)} distinct words)")
        else:
            V_train = V
        keep = mask.any(1)
        colmask = None
        if a.holdout_frac > 0:
            rng = np.random.default_rng(1234)
            hold = np.nonzero(rng.random(len(doc_ids)) < a.holdout_frac)[0]
            write_json(ck_dir / "holdout.json", [int(r) for r in hold])
            keep &= ~np.isin([x["src"] for x in train], hold)
            colmask = torch.zeros(len(doc_ids), dtype=torch.bool, device=DEV)
            colmask[torch.tensor(hold, device=DEV)] = True
            print(f"holdout: {len(hold)} docs hidden from training (experiment run — do not upload these assets)")
        B = to_batch(idx[keep], mask[keep], fb[keep])
        T = T_all[torch.tensor(np.nonzero(keep)[0], device=DEV)]
        src = torch.tensor([x["src"] for x, k in zip(train, keep) if k], device=DEV)
        print(f"train: {int(keep.sum())}/{len(train)} queries with >=1 slot · {DEV} · cfg {cfg}")
        idf_core = np.array([idf(t) for t in terms], np.float32)
        ext_init = float(np.log(len(doc_ids) + 1) + 1.0)  # IDF of a word no document contains
        for s in range(a.seeds):
            t0 = time.time()
            model = Student(V_train, idf_core, D.shape[1], len(terms), ext_init).to(DEV)
            model, val, eps = train_one(model, B, T, src, D, cfg, seed=s, colmask=colmask)
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in PARAM_KEYS}
            torch.save({"state_dict": sd, "seed": s, "config": cfg, "val_loss": val, "epochs": eps,
                        "fingerprint": fingerprint, "basis_sha1": basis_sha1, "teacher": a.teacher},
                       ck_dir / f"seed{s}.pt")
            states.append(sd)
            print(f"  seed {s}: val loss {val:.4f}, {eps} epochs, ext weight {float(F.softplus(sd['extw'])):.3f}, "
                  f"{time.time() - t0:.0f}s -> checkpoints/seed{s}.pt")

    members = [{
        "term_scale": Vs * F.softplus(sd["theta"]).numpy(),
        "proj_w": sd["lin.weight"].numpy(),
        "proj_b": sd["lin.bias"].numpy(),
        "fb_weight": float(torch.exp(sd["fbw"])),
        # checkpoints from before the lexicon have no extw: lexicon slots then carry no weight
        "ext_weight": float(F.softplus(sd["extw"])) if "extw" in sd and lex is not None else 0.0,
    } for sd in states]
    sizes = write_assets(out, terms=terms, vocab_q=Vq, members=members, docs=D_np, doc_ids=doc_ids,
                         teacher=a.teacher, config=cfg,
                         extra={"trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "input_pca_explained": float(vin["explained"]),
                                "basis_sha1": basis_sha1 if lex is not None else "", **fingerprint})
    total = sum(sizes.values())
    print(f"assets -> {out}: " + ", ".join(f"{k} {v / 1024:.0f}KB" for k, v in sizes.items()) + f" (total {total / 1e6:.2f}MB)")

    # Quick check that the shipped format scores like the float model (full report: evaluate.py)
    ev_path = work / "eval_sets.json"
    if ev_path.exists():
        ev = load_json(ev_path)
        rt = StudentRuntime(out, lexicon_dir=a.index)
        for k in ("dev", "test", "oov"):
            if not ev.get(k):
                continue
            Q, valid, _ = rt.encode(ev[k])
            m = metrics(ranks(Q, valid, ev[k], rt.docs))
            print(f"  deployed {k}: MRR {m['mrr']:.3f} · R@10 {m['r10']:.1f} (n={m['n']})")


if __name__ == "__main__":
    main()
