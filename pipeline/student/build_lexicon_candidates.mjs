/*
	Extension lexicon — step A: out-of-corpus candidate words (run rarely; see build_lexicon.py).

	The student only knows words that occur in the corpus, so a query built around a word no
	article contains maps to no slot at all. This collects general Korean nouns from two public
	sources (downloaded here, never redistributed — licenses in docs/STUDENT.md):
	  · OpenSubtitles 2018 Korean frequency list (colloquial; hermitdave/FrequencyWords)
	  · Korean Wikipedia article titles (neologisms and concepts: 내로남불, 헬조선, 갑질 …)
	and reduces them with garu (the runtime tokenizer) to:
	  sub      [[noun, freq]]  nouns of subtitle eojeols, hangul only, >= 2 chars, freq >= 2
	  single   wiki titles garu reads as exactly one noun (hangul, 2..6 chars)
	  compound wiki titles garu splits (hangul, 2..4 chars) — reachable via eojeol tokens

	Run: node pipeline/student/build_lexicon_candidates.mjs [--src data/student/lexicon/src --out data/student/lexicon/candidates.json]
*/
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { Garu } from 'garu-ko';

const LEXDIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'data', 'student', 'lexicon');
const args = {
	src: join(LEXDIR, 'src'),
	out: join(LEXDIR, 'candidates.json'),
	...Object.fromEntries(
		process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), [])
	)
};

const SOURCES = {
	subtitles: 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ko/ko_full.txt',
	wikiTitles: 'https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-all-titles-in-ns0.gz'
};
const SUB_MIN_FREQ = 2;
const HANGUL = /^[가-힣]+$/;

mkdirSync(args.src, { recursive: true });
async function fetchOnce(url, file) {
	const path = join(args.src, file);
	if (!existsSync(path)) {
		const r = await fetch(url);
		if (!r.ok) throw new Error(`${r.status} ${url}`);
		writeFileSync(path, Buffer.from(await r.arrayBuffer()));
		console.log('downloaded', file);
	}
	return readFileSync(path);
}

const garu = await Garu.load();

const sub = new Map();
for (const line of (await fetchOnce(SOURCES.subtitles, 'ko_full.txt')).toString('utf-8').split('\n')) {
	const [tok, f] = line.trim().split(' ');
	if (!tok) continue;
	for (const w of garu.nouns(tok, { includeSL: false })) {
		if (w.length >= 2 && HANGUL.test(w)) sub.set(w, (sub.get(w) || 0) + Number(f || 0));
	}
}

const single = [];
const compound = [];
const titles = gunzipSync(await fetchOnce(SOURCES.wikiTitles, 'kowiki-titles.gz')).toString('utf-8').split('\n');
for (const t of titles) {
	if (!HANGUL.test(t) || t.length < 2 || t.length > 6) continue;
	const ns = garu.nouns(t, { includeSL: false });
	if (ns.length === 1 && ns[0] === t) single.push(t);
	else if (t.length <= 4) compound.push(t);
}
garu.destroy();

const subList = [...sub].filter(([, f]) => f >= SUB_MIN_FREQ).sort((a, b) => b[1] - a[1]);
mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, JSON.stringify({ sources: SOURCES, sub: subList, single, compound }));
console.log(`candidates: subtitle nouns ${subList.length} · wiki single ${single.length} · wiki compound ${compound.length}`);
