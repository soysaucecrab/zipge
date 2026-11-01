/*
	Student encoder — step 1/4: vocabulary, training queries and eval sets.

	Inputs
	  --articles DIR   article JSONs ({id, title, content}, docs/USAGE.md §2-1)   default data/articles
	  --index DIR      search index holding doc_meta.json (build_index.py) — its row order is the
	                   row order of every doc-vector file                          default data/index
	  --work DIR       output dir                                                  default data/student
	  --eval FILE      optional hand-written queries [{text, relevant: [ids]}] — the ones the keyword
	                   tier cannot already catch are split into dev (even) / test (odd)
	  --oov FILE       optional queries built around words absent from the corpus (same shape)
	Outputs (work dir)
	  vocab.json          { n_docs, terms, df }  terms = nouns with 2 <= df <= 60% of docs
	  train_queries.json  [{ text, toks, kinds, src, kind }]  src = source doc row (-1 = none)
	  eval_sets.json      { dev, test, title, bodyseg, oov }: [{ text, toks, kinds, rows, covered }]

	Run: node pipeline/student/build_queries.mjs [--eval my_queries.json]
*/
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Garu } from 'garu-ko';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = {
	articles: join(ROOT, 'data', 'articles'),
	index: join(ROOT, 'data', 'index'),
	work: join(ROOT, 'data', 'student'),
	...Object.fromEntries(
		process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), [])
	)
};

const BODY_CHARS = 8000; // body prefix used for vocabulary, windows and segments
const WINDOWS_PER_DOC = 40; // random 2..5 consecutive-noun windows per article
const SEGMENTS_PER_DOC = 20; // random 8..40-char body segments per article
const NEAR_DUP_JACCARD = 0.6; // training queries this close to any eval query are dropped
const VOCAB_MIN_DF = 2;
const VOCAB_MAX_DF_RATIO = 0.6;
const TITLE_EVAL = 120;

const garu = await Garu.load();
const nounsOf = (s) => garu.nouns(String(s || ''), { includeSL: true }).map((n) => n.toLowerCase());

// Must stay identical to src/korean.ts (PARTICLES + stripParticle): the runtime tokenises queries the same way.
const PARTICLES = [
	'에서부터', '으로부터', '이라도', '에게서', '으로서', '으로써', '까지', '부터', '처럼', '보다', '에게',
	'한테', '께서', '이나', '이란', '라는', '든지', '마다', '조차', '밖에', '이랑', '하고', '은', '는', '이',
	'가', '을', '를', '과', '와', '도', '만', '의', '에', '로', '랑', '라', '뿐'
].sort((a, b) => b.length - a.length);
function stripParticle(word) {
	if (!/[가-힣]/.test(word)) return word;
	for (const p of PARTICLES) {
		if (word.length - p.length >= 2 && word.endsWith(p)) return word.slice(0, -p.length);
	}
	return word;
}

// src/korean.ts studentTokens: n = nouns, e = eojeol (+ particle-stripped), j = joined query
function tokenize(text) {
	const cleaned = text.trim();
	const lower = cleaned.toLowerCase();
	const nouns = nounsOf(cleaned);
	const out = [];
	const kinds = [];
	const add = (t, k) => {
		if (t && !out.includes(t)) {
			out.push(t);
			kinds.push(k);
		}
	};
	for (const t of nouns.length ? nouns : lower.split(/\s+/)) add(t, 'n');
	for (const e of lower.split(/\s+/)) {
		if (e.length < 2) continue;
		add(e, 'e');
		add(stripParticle(e), 'e');
	}
	add(lower.replace(/\s+/g, ''), 'j');
	return { toks: out, kinds };
}

const meta = JSON.parse(readFileSync(join(args.index, 'doc_meta.json'), 'utf-8'));
const idToRow = new Map(meta.map((m, i) => [String(m.id), i]));
const titleTerms = meta.map((m) => new Set(nounsOf(m.title)));
const titleJoined = meta.map((m) => String(m.title).replace(/\s+/g, '').toLowerCase());

// Keyword-tier proxy: a query is "covered" when its nouns mostly appear in the doc title
function kwCovers(text, row) {
	const qn = nounsOf(text);
	const jt = titleJoined[row];
	const jq = text.replace(/\s+/g, '').toLowerCase();
	if (jq.length >= 2 && jt.includes(jq)) return true;
	if (!qn.length) return false;
	const tt = titleTerms[row];
	return qn.filter((t) => tt.has(t) || jt.includes(t)).length / qn.length >= 0.6;
}
const item = (text, ids) => {
	const rows = ids.map((id) => idToRow.get(String(id))).filter((r) => r !== undefined);
	return { text, ...tokenize(text), rows, covered: rows.some((r) => kwCovers(text, r)) };
};

// Deterministic RNG so reruns produce the same query set for the same corpus
let seed = 77;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

const byId = new Map();
for (const f of readdirSync(args.articles)) {
	if (!f.endsWith('.json')) continue;
	const d = JSON.parse(readFileSync(join(args.articles, f), 'utf-8'));
	byId.set(String(d.id), d);
}
const bodyText = (content) =>
	String(content || '')
		.replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
		.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
		.replace(/<\/(p|div|li|h\d)>|<br\s*\/?>/gi, '\n')
		.replace(/<[^>]+>/g, ' ')
		.replace(/&[a-z#0-9]+;/gi, ' ')
		.slice(0, BODY_CHARS);
const cleanTitle = (t) =>
	String(t)
		.replace(/\[[^\]]*\]/g, ' ')
		.replace(/\([^)]*\)/g, ' ')
		.replace(/[<>『』"“”]/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

const df = new Map();
const train = [];
const bodyseg = [];
const seen = new Set();
const push = (text, src, kind) => {
	text = text.replace(/\s+/g, ' ').trim();
	if (text.length < 2 || text.length > 60 || !/[가-힣]/.test(text) || seen.has(text)) return;
	seen.add(text);
	train.push({ text, ...tokenize(text), src, kind });
};

let missing = 0;
for (const [row, m] of meta.entries()) {
	const a = byId.get(String(m.id));
	if (!a) {
		missing++;
		continue;
	}
	const body = bodyText(a.content);
	const nouns = nounsOf(a.title + ' ' + body.replace(/\s+/g, ' ')).filter((w) => w.length >= 2);
	for (const w of new Set(nouns)) df.set(w, (df.get(w) || 0) + 1);

	let segs = body
		.split(/[.!?\n,;:·…"“”'‘’()[\]]/)
		.map((s) => s.replace(/\s+/g, ' ').trim())
		.filter((s) => s.length >= 8 && s.length <= 40 && /[가-힣]{2}/.test(s));
	segs = [...new Set(segs)];
	// author-independent eval: one held-out body segment per article -> its own article
	const cands = segs.filter((s) => s.length >= 15 && !kwCovers(s, row));
	if (cands.length) {
		const pick = cands[Math.floor(rnd() * cands.length)];
		bodyseg.push({ text: pick, ...tokenize(pick), rows: [row], covered: false });
		segs = segs.filter((s) => s !== pick);
		seen.add(pick);
	}
	for (let k = 0; k < WINDOWS_PER_DOC && nouns.length > 3; k++) {
		const len = 2 + Math.floor(rnd() * 4);
		const st = Math.floor(rnd() * Math.max(1, nouns.length - len));
		push(nouns.slice(st, st + len).join(' '), row, 'win');
	}
	for (let k = 0; k < SEGMENTS_PER_DOC && segs.length; k++) push(segs[Math.floor(rnd() * segs.length)], row, 'seg');
}
for (const [row, m] of meta.entries()) {
	for (const part of String(m.title).split('·')) push(cleanTitle(part), row, 'title');
	for (const t of m.tags || []) push(t, -1, 'tag');
}

// ---- eval sets
const readEval = (file) => (file ? JSON.parse(readFileSync(file, 'utf-8')) : []);
const concept = readEval(args.eval).map((q) => item(q.text, q.relevant)).filter((q) => q.rows.length);
const open = concept.filter((q) => !q.covered); // keyword-covered queries say nothing about semantics
const dev = open.filter((_, i) => i % 2 === 0);
const test = open.filter((_, i) => i % 2 === 1);
const oov = readEval(args.oov).map((q) => item(q.text, q.relevant)).filter((q) => q.rows.length);
// title -> its own article, for a deterministic sample of documents
const titleRows = meta.map((_, i) => i);
for (let i = titleRows.length - 1; i > 0; i--) {
	const j = Math.floor(rnd() * (i + 1));
	[titleRows[i], titleRows[j]] = [titleRows[j], titleRows[i]];
}
titleRows.length = Math.min(titleRows.length, TITLE_EVAL);
const title = titleRows
	.map((row) => ({ row, text: cleanTitle(meta[row].title) }))
	.filter(({ text }) => text.length >= 2 && /[가-힣]/.test(text))
	.map(({ row, text }) => ({ text, ...tokenize(text), rows: [row], covered: true }));
const evalSets = { dev, test, title, bodyseg, oov };

// ---- near-duplicate removal against every eval query
const evalAll = Object.values(evalSets).flat();
const inv = new Map();
evalAll.forEach((q, i) => q.toks.forEach((t) => (inv.get(t) || inv.set(t, []).get(t)).push(i)));
const evalTokSets = evalAll.map((q) => new Set(q.toks));
const evalTexts = new Set(evalAll.map((q) => q.text));
let dropped = 0;
const trainClean = train.filter((q) => {
	if (evalTexts.has(q.text)) return (dropped++, false);
	const s = new Set(q.toks);
	const cand = new Set();
	for (const t of s) for (const i of inv.get(t) || []) cand.add(i);
	for (const i of cand) {
		const e = evalTokSets[i];
		let inter = 0;
		for (const t of s) if (e.has(t)) inter++;
		if (inter / (s.size + e.size - inter) >= NEAR_DUP_JACCARD) return (dropped++, false);
	}
	return true;
});

// ---- vocabulary
const nDocs = meta.length - missing;
const terms = [...df.entries()]
	.filter(([, c]) => c >= VOCAB_MIN_DF && c <= nDocs * VOCAB_MAX_DF_RATIO)
	.map(([w]) => w)
	.sort();

mkdirSync(args.work, { recursive: true });
writeFileSync(join(args.work, 'vocab.json'), JSON.stringify({ n_docs: nDocs, terms, df: Object.fromEntries(df) }));
writeFileSync(join(args.work, 'train_queries.json'), JSON.stringify(trainClean));
writeFileSync(join(args.work, 'eval_sets.json'), JSON.stringify(evalSets));
garu.destroy();

const byKind = trainClean.reduce((acc, q) => ((acc[q.kind] = (acc[q.kind] || 0) + 1), acc), {});
console.log(`docs ${nDocs} (missing articles ${missing}) · vocab ${terms.length} terms (df table ${df.size})`);
console.log(`train ${trainClean.length} (near-dup dropped ${dropped}) ${JSON.stringify(byKind)}`);
console.log(
	`eval: user queries ${concept.length} -> keyword-covered ${concept.length - open.length} -> dev ${dev.length} / test ${test.length} · title ${title.length} · bodyseg ${bodyseg.length} · oov ${oov.length}`
);
