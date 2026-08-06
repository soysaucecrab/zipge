/*
	락다운(WASM 차단) 폴백용 사전 산출 — garu를 오프라인(node)에서 돌려
	브라우저가 하던 문서 토큰화를 미리 해 둔다.
	입력: static/search-index/doc_meta.json
	출력: static/search-index/terms.json
	  { docs: [{ t: [제목 단어들], r: [태그·저자 단어들] }, ...] }  (meta와 같은 순서)
	Run: node pipeline/build_terms.js (ESM)   (이 레포 루트에서; garu-ko는 devDependency)
*/
import { readFileSync, writeFileSync } from 'node:fs';
import { Garu } from 'garu-ko';

async function main() {
	const meta = JSON.parse(readFileSync('static/search-index/doc_meta.json', 'utf-8'));
	const garu = await Garu.load();
	const nouns = (text) => {
		const trimmed = (text || '').trim();
		if (!trimmed) return [];
		const out = garu.nouns(trimmed, { includeSL: true });
		return (out.length > 0 ? out : trimmed.split(/\s+/)).map((t) => t.toLowerCase());
	};

	const docs = meta.map((m) => ({
		t: nouns(m.title),
		r: [...m.tags, ...m.authors.map((a) => a.name)].flatMap(nouns)
	}));
	garu.destroy();

	writeFileSync('static/search-index/terms.json', JSON.stringify({ docs }));
	const size = JSON.stringify({ docs }).length;
	console.log(`terms.json: ${meta.length} docs, ${(size / 1024).toFixed(0)}KB`);
}

main();
