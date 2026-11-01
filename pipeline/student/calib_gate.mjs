/*
	Calibrate the student relevance gate (SearchController option studentGate) on your corpus.
	The scoring is a line-for-line port of src/core.ts studentQuery/studentSearch, so the
	thresholds measured here transfer to the browser. No model runs — lookups and one projection.

	Run (repo root, after pipeline/student/run.sh):
	  node pipeline/student/calib_gate.mjs [--index data/index] [--eval queries.json] [--oov oov.json]
	                                       [--offdomain probes.json]
	  --eval / --oov   [{text, relevant: [ids]}] — queries that must keep their answers
	  --offdomain      ["query", ...] — real questions your corpus cannot answer
	                   (a food blog's off-domain list differs from an economics site's)
	  PARITY='q1|q2' node ... prints the student top-6 per query (browser parity check)

	Gate: a query passes when the top cosine >= exact (some token hit the vocabulary or the
	extension lexicon exactly) or >= fallbackOnly (only substring-fallback slots). fallbackOnly is
	set just above every fallback-only probe; exact stays 0.3 unless a query whose answer is in the
	top 10 would be rejected.
*/
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Garu } from 'garu-ko';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = {
	index: join(ROOT, 'data', 'index'),
	...Object.fromEntries(
		process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), [])
	)
};
const garu = await Garu.load();
const J = (p) => JSON.parse(readFileSync(p, 'utf-8'));
const B = (p) => {
	const b = readFileSync(p);
	return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const dir = args.index;

// ---- assets (src/core.ts loadStudent)
const meta = J(join(dir, 'doc_meta.json'));
const m = J(join(dir, 'student_meta.json'));
const V = m.vocab_size, dim = m.dim, out = m.out_dim, M = m.members;
const tail = (m.version ?? 1) >= 2 ? 2 : 1, block = V + 2 * out + tail;
const p = new Float32Array(B(join(dir, 'student_params_f32.bin')));
const ds = J(join(dir, 'student_doc_scales.json'));
const metaRow = new Map(meta.map((d, i) => [String(d.id), i]));
const members = [...Array(M).keys()];
const S = {
	index: new Map(m.terms.map((t, i) => [t, i])),
	maxSlots: m.max_slots ?? 16, fbMinLen: m.fallback?.min_len ?? 3, fbMinRatio: m.fallback?.min_ratio ?? 0.5,
	vocab: new Int8Array(B(join(dir, 'student_vocab_i8.bin'))),
	proj: new Int8Array(B(join(dir, 'student_proj_i8.bin'))),
	termScale: members.map((k) => p.subarray(k * block, k * block + V)),
	projScale: members.map((k) => p.subarray(k * block + V, k * block + V + out)),
	projBias: members.map((k) => p.subarray(k * block + V + out, k * block + V + 2 * out)),
	fbWeight: members.map((k) => p[k * block + V + 2 * out]),
	extWeight: members.map((k) => (tail === 2 ? p[k * block + V + 2 * out + 1] : 0)),
	docs: new Int8Array(B(join(dir, 'student_doc_vectors_i8.bin'))),
	docScales: new Float32Array(ds.scales),
	docRow: Int32Array.from(m.doc_ids, (id) => metaRow.get(String(id)) ?? -1)
};

// ---- extension lexicon: every shard here; the browser Range-fetches only the ones it needs
const LEX = new Map();
try {
	const lm = J(join(dir, 'student_lex_meta.json'));
	if (tail === 2 && m.basis_sha1 && lm.basis_sha1 === m.basis_sha1) {
		const packs = lm.packs.map((name) => B(join(dir, name)));
		for (const [pk, start, length] of lm.index) {
			const buf = packs[pk].slice(start, start + length);
			const dv = new DataView(buf);
			const count = dv.getUint32(0, true), jl = dv.getUint32(4, true);
			const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jl)));
			const q = new Int8Array(buf, 8 + jl, count * dim);
			head.terms.forEach((t, r) => LEX.set(t, { q, off: r * dim, scale: head.scales[r] }));
		}
	}
} catch {
	/* no lexicon */
}

// ---- src/korean.ts studentTokens
const PARTICLES = ['에서부터', '으로부터', '이라도', '에게서', '으로서', '으로써', '까지', '부터', '처럼', '보다', '에게', '한테', '께서', '이나', '이란', '라는', '든지', '마다', '조차', '밖에', '이랑', '하고', '은', '는', '이', '가', '을', '를', '과', '와', '도', '만', '의', '에', '로', '랑', '라', '뿐'].sort((a, b) => b.length - a.length);
function stripParticle(w) {
	if (!/[가-힣]/.test(w)) return w;
	for (const x of PARTICLES) if (w.length - x.length >= 2 && w.endsWith(x)) return w.slice(0, -x.length);
	return w;
}
function studentTokens(text) {
	const cleaned = text.trim();
	if (!cleaned) return [];
	const lower = cleaned.toLowerCase();
	const res = [];
	const add = (t, kind) => { if (t && !res.some((x) => x.t === t)) res.push({ t, kind }); };
	const nouns = garu.nouns(cleaned, { includeSL: true }).map((n) => n.toLowerCase());
	for (const t of nouns.length > 0 ? nouns : lower.split(/\s+/)) add(t, 'n');
	for (const e of lower.split(/\s+/)) { if (e.length < 2) continue; add(e, 'e'); add(stripParticle(e), 'e'); }
	add(lower.replace(/\s+/g, ''), 'j');
	return res;
}

// ---- src/core.ts studentFallback / studentQuery / studentSearch
function studentFallback(t) {
	for (let L = t.length - 1; L >= 2 && L / t.length >= S.fbMinRatio; L--)
		for (let i = 0; i + L <= t.length; i++) { const r = S.index.get(t.slice(i, i + L)); if (r !== undefined) return r; }
	return undefined;
}
function studentQuery(text) {
	const slots = [], used = new Set(), c = { core: 0, lexicon: 0, fallback: 0 };
	for (const { t, kind } of studentTokens(text)) {
		if (slots.length >= S.maxSlots) break;
		let slot = null;
		const row = S.index.get(t);
		if (row !== undefined) slot = { key: `c${row}`, row, fb: false };
		else if (LEX.has(t)) { const e = LEX.get(t); slot = { key: `x${t}`, q: e.q, off: e.off, scale: e.scale }; }
		else if (kind === 'n' && t.length >= S.fbMinLen) { const fr = studentFallback(t); if (fr !== undefined) slot = { key: `c${fr}`, row: fr, fb: true }; }
		if (!slot || used.has(slot.key)) continue;
		used.add(slot.key);
		slots.push(slot);
		if (slot.q) c.lexicon++; else if (slot.fb) c.fallback++; else c.core++;
	}
	if (!slots.length) return null;
	const pooled = new Float32Array(dim), member = new Float32Array(out), q = new Float32Array(out);
	for (let k = 0; k < M; k++) {
		pooled.fill(0);
		for (const sl of slots) {
			const src = sl.q ? sl.q : S.vocab, off = sl.q ? sl.off : sl.row * dim;
			const w = sl.q ? sl.scale * S.extWeight[k] : S.termScale[k][sl.row] * (sl.fb ? S.fbWeight[k] : 1);
			for (let d = 0; d < dim; d++) pooled[d] += src[off + d] * w;
		}
		const base = k * out * dim;
		let n = 0;
		for (let o = 0; o < out; o++) {
			let acc = 0;
			const off = base + o * dim;
			for (let d = 0; d < dim; d++) acc += S.proj[off + d] * pooled[d];
			const v = S.projScale[k][o] * acc + S.projBias[k][o];
			member[o] = v;
			n += v * v;
		}
		n = Math.sqrt(n) || 1;
		for (let o = 0; o < out; o++) q[o] += member[o] / n;
	}
	let n = 0;
	for (let o = 0; o < out; o++) n += q[o] * q[o];
	n = Math.sqrt(n) || 1;
	for (let o = 0; o < out; o++) q[o] /= n;
	return { q, slots: c };
}
function studentSearch(q) {
	const rows = [], vals = [];
	for (let i = 0; i < S.docRow.length; i++) {
		const r = S.docRow[i];
		if (r < 0) continue;
		let dot = 0;
		const off = i * out;
		for (let o = 0; o < out; o++) dot += S.docs[off + o] * q[o];
		rows.push(r);
		vals.push(dot * S.docScales[i]);
	}
	const order = vals.map((_, j) => j).sort((a, b) => vals[b] - vals[a]);
	return { top: vals[order[0]], rows: order.map((j) => rows[j]), scores: order.map((j) => vals[j]) };
}

if (process.env.PARITY) {
	for (const text of process.env.PARITY.split('|')) {
		const enc = studentQuery(text);
		if (!enc) { console.log(`[${text}] no slots`); continue; }
		const r = studentSearch(enc.q);
		console.log(`[${text}] slots core ${enc.slots.core} lex ${enc.slots.lexicon} fb ${enc.slots.fallback} · top ${r.top.toFixed(3)}`);
		r.rows.slice(0, 6).forEach((row, k) => console.log(`  ${k + 1}. ${r.scores[k].toFixed(3)} ${meta[row].id} ${String(meta[row].title).slice(0, 50)}`));
	}
	process.exit(0);
}

// Gibberish that must not surface results: hangul/latin mashes, jamo, syllable runs that may hit the fallback
const GARBAGE = ['쀍쀍쀍없는글', 'ㅁㄴㅇㄹㅁㄴㅇㄹ', 'asdf qwer zxcv', 'ㅋㅋㅋㅋㅋㅋㅋ', '123456789', 'qwertyuiop', 'ㄱㄴㄷㄹㅁㅂㅅ',
	'블라블라블라블라', 'ㅇㅇㅇㅇㅇㅇ', 'zzzzzz난수문자열', '궭뷁쓁', '하하하하하하하', 'lorem ipsum dolor', 'xkcd zzz qqq',
	'a1b2c3 d4e5', 'asdfghjkl', 'hjkl yuio', '가나다라마바사', '아야어여오요우유', '갸냐댜랴먀뱌', '뷁뷁 쀍쀍', '퉤퉤퉤', '샬라샬라'];
const cleanTitle = (t) => String(t).replace(/\[[^\]]*\]/g, ' ').replace(/\([^)]*\)/g, ' ').replace(/\s+/g, ' ').trim();
const readList = (file) => (file ? J(file) : []);
const sets = {
	title: meta.slice(0, 200).map((d) => ({ text: cleanTitle(d.title), rel: [d.id] })).filter((x) => x.text.length >= 2),
	eval: readList(args.eval).map((x) => ({ text: x.text, rel: x.relevant })),
	oov: readList(args.oov).map((x) => ({ text: x.text, rel: x.relevant })),
	garbage: GARBAGE.map((t) => ({ text: t, rel: null })),
	offdomain: readList(args.offdomain).map((t) => ({ text: t, rel: null }))
};
const R = {};
for (const [k, items] of Object.entries(sets)) {
	R[k] = items.map(({ text, rel }) => {
		const enc = studentQuery(text);
		if (!enc) return { text, none: true };
		const r = studentSearch(enc.q);
		let rank = Infinity;
		if (rel) {
			const relRows = new Set(rel.map((id) => metaRow.get(String(id))).filter((x) => x !== undefined));
			const pos = r.rows.findIndex((row) => relRows.has(row));
			rank = pos < 0 ? Infinity : pos;
		}
		const cls = enc.slots.core + enc.slots.lexicon > 0 ? 'exact' : 'fb';
		return { text, top: r.top, rank, cls };
	});
}
const pct = (a, q) => { const s2 = [...a].sort((x, y) => x - y); return s2.length ? s2[Math.floor(q * (s2.length - 1))] : NaN; };
console.log(`student assets: ${m.terms.length} terms · ${M} members · lexicon ${LEX.size} words · docs matched ${[...S.docRow].filter((r) => r >= 0).length}/${S.docRow.length}`);
for (const [k, rows] of Object.entries(R)) {
	if (!rows.length) continue;
	const ok = rows.filter((r) => !r.none);
	console.log(`${k.padEnd(9)} n=${rows.length} no-slot=${rows.length - ok.length} exact=${ok.filter((r) => r.cls === 'exact').length} fallback-only=${ok.filter((r) => r.cls === 'fb').length} · top p5/50/95=${[0.05, 0.5, 0.95].map((q) => pct(ok.map((r) => r.top), q).toFixed(3)).join('/')}`);
}

const probes = [...R.garbage, ...R.offdomain].filter((r) => !r.none && r.cls === 'fb');
const fallbackOnly = probes.length ? +(Math.max(...probes.map((r) => r.top)) + 0.01).toFixed(3) : 0.635;
const relevant = [...R.title, ...R.eval, ...R.oov].filter((r) => !r.none && r.cls === 'exact' && r.rank < 10);
const exact = +Math.min(0.3, ...relevant.map((r) => r.top - 0.005)).toFixed(3);
const G = { exact, fallbackOnly };
const pass = (r) => !r.none && r.top >= (r.cls === 'fb' ? G.fallbackOnly : G.exact);
console.log(`\nfallback-only probes: ${probes.length}${probes.length ? '' : ' (none — keeping the default)'}`);
console.log('proposed studentGate =', JSON.stringify(G));
for (const [k, rows] of Object.entries(R)) if (rows.length) console.log(`  pass ${k.padEnd(9)} ${rows.filter(pass).length}/${rows.length}`);
for (const k of ['title', 'eval', 'oov']) {
	const lost = R[k].filter((r) => !pass(r) && !r.none && r.rank < 10);
	if (R[k].length) console.log(`  ${k}: rejected although the answer was in the top 10: ${lost.length}${lost.length ? ' — ' + lost.map((r) => r.text).join(', ') : ''}`);
}
garu.destroy();
