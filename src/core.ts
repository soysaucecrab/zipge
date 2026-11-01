
import { extractTerms, loadAnalyzer, studentTokens, termsMatch } from './korean.js';

export type SearchHit = {
	id: string;
	slug: string;
	title: string;
	board: string | null;
	boardName: string | null;
	category: string | null;
	tags: string[];
	authors: { name: string; role: string }[];
	created: string;
	thumbnail: string | null;
	preview: string | null;
	views: number;
	score: number;
	source: 'keyword' | 'semantic';
};

type DocMeta = Omit<SearchHit, 'score' | 'source'>;

// Student encoder assets, layout in docs/STUDENT.md
type StudentAssets = {
	index: Map<string, number>;
	dim: number;
	outDim: number;
	members: number;
	maxSlots: number;
	fbMinLen: number;
	fbMinRatio: number;
	vocab: Int8Array; // [V][dim]
	proj: Int8Array; // [members][outDim][dim]
	termScale: Float32Array[]; // per member [V]
	projScale: Float32Array[]; // per member [outDim]
	projBias: Float32Array[]; // per member [outDim]
	fbWeight: number[];
	extWeight: number[]; // 0 = no usable extension lexicon
	docs: Int8Array; // [N][outDim]
	docScales: Float32Array;
	docRow: Int32Array; // student doc row -> doc_meta row, -1 when absent
	lex: { packs: string[]; index: [number, number, number][] } | null; // per shard [pack, start, length]
};

type LexShard = { index: Map<string, number>; q: Int8Array; scales: Float32Array };

export type StudentEncoding = { vector: Float32Array; core: number; lexicon: number; fallback: number };

// FNV-1a 32-bit over UTF-16 code units (pipeline/student/common.py fnv1a_utf16)
function fnv1a(s: string): number {
	let h = 0x811c9dc5;
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i);
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h >>> 0;
}

const okJson = (r: Response) => {
	if (!r.ok) throw new Error(`${r.status} ${r.url}`);
	return r.json();
};
const okBuf = (r: Response) => {
	if (!r.ok) throw new Error(`${r.status} ${r.url}`);
	return r.arrayBuffer();
};

export class SearchIndex {
	private base: string;
	private meta: DocMeta[] | null = null;
	private metaLoading: Promise<void> | null = null;

	constructor(base = '/search-index') {
		this.base = base;
	}

	private tagCounts = new Map<string, number>();

	loadMeta(): Promise<void> {
		this.metaLoading ??= fetch(`${this.base}/doc_meta.json`)
			.then((r) => r.json())
			.then((meta: DocMeta[]) => {
				this.meta = meta;
				for (const m of meta)
					for (const t of m.tags) this.tagCounts.set(t, (this.tagCounts.get(t) ?? 0) + 1);
			});
		return this.metaLoading;
	}

	matchTags(terms: string[], k = 3): { name: string; count: number }[] {
		const scored: { name: string; count: number; score: number }[] = [];
		for (const [name, count] of this.tagCounts) {
			let best = 0;
			for (const t of terms) best = Math.max(best, termsMatch(t, name.toLowerCase()));
			if (best >= 0.65) scored.push({ name, count, score: best });
		}
		return scored
			.sort((a, b) => b.score - a.score || b.count - a.count)
			.slice(0, k)
			.map(({ name, count }) => ({ name, count }));
	}

	private docTerms: { title: string[]; rest: string[] }[] | null = null;
	private vocab: string[] = [];

	async prepareKeyword(): Promise<void> {
		await this.loadMeta();
		if (this.docTerms) return;
		if (await loadAnalyzer()) {
			this.docTerms = this.meta!.map((m) => ({
				title: extractTerms(m.title),
				rest: [...m.tags, ...m.authors.map((a) => a.name)].flatMap(extractTerms)
			}));
		} else {
			try {
				const t = await fetch(`${this.base}/terms.json`).then((r) => r.json());
				this.docTerms = (t.docs as { t: string[]; r: string[] }[]).map((d) => ({
					title: d.t,
					rest: d.r
				}));
			} catch {
				return; // 사전도 못 받으면 부분 문자열 폴백으로만 동작
			}
		}
		const single = this.docTerms.flatMap((d) => [...d.title, ...d.rest]);
		const bigrams = this.docTerms.flatMap((d) =>
			d.title.slice(0, -1).map((t, i) => t + d.title[i + 1])
		);
		this.vocab = [...new Set([...single, ...bigrams])];
	}

	private correct(interpretations: string[][], original: string): string | null {
		if (this.vocab.length === 0) return null;
		const originalLen = original.replace(/\s+/g, '').length;
		let best: {
			terms: string[];
			quality: number;
			charDiff: number;
			termCount: number;
			replaced: boolean;
		} | null = null;
		for (const interp of interpretations) {
			const corrected: string[] = [];
			let total = 0;
			let replaced = false;
			for (const term of interp) {
				let bestScore = 0;
				let bestTerm = term;
				for (const v of this.vocab) {
					const s = termsMatch(term, v);
					// 동점이면 길이가 쿼리에 가까운 후보 — "사유재솬"에 "자유"보다 "사유재산"
					const closer =
						s === bestScore &&
						s > 0 &&
						Math.abs(v.length - term.length) < Math.abs(bestTerm.length - term.length);
					if (s > bestScore || closer) {
						bestScore = s;
						bestTerm = v;
						if (s === 1) break;
					}
				}
				if (bestScore === 0) {
					total = 0;
					break;
				}
				total += bestScore;
				if (bestScore < 1 && bestScore >= 0.65 && bestTerm !== term) {
					corrected.push(bestTerm);
					replaced = true;
				} else {
					corrected.push(term);
				}
			}
			if (total === 0) continue;
			const quality = total / interp.length;
			const charDiff = Math.abs(corrected.join('').length - originalLen);
			const better =
				!best ||
				quality > best.quality ||
				(quality === best.quality &&
					(charDiff < best.charDiff ||
						(charDiff === best.charDiff && interp.length < best.termCount)));
			if (better)
				best = { terms: corrected, quality, charDiff, termCount: interp.length, replaced };
		}
		if (!best) return null;
		const correctedQuery = best.terms.join(' ');
		if (correctedQuery === original) return null;
		const sameChars = correctedQuery.replace(/\s+/g, '') === original.replace(/\s+/g, '');
		const mergedSpacing =
			sameChars && best.terms.length < original.split(/\s+/).filter(Boolean).length;
		if (!best.replaced && !mergedSpacing) return null;
		return correctedQuery;
	}

	lastCorrection: string | null = null;

	keywordSearch(query: string, k = 20): SearchHit[] {
		this.lastCorrection = null;
		if (!this.meta) return [];
		const rawTerms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
		let nounTerms = extractTerms(query);
		const queryChars = query.replace(/\s+/g, '').length;
		if (rawTerms.length === 1 && nounTerms.join('').length < queryChars * 0.7) nounTerms = [];
		const joined = query.trim().toLowerCase().replace(/\s+/g, '');
		const termSets = [nounTerms, rawTerms].filter((s) => s.length > 0);
		if (rawTerms.length > 1 && joined.length >= 2) termSets.push([joined]);
		if (termSets.length === 0) return [];

		const phrase = rawTerms.join(' ');
		const literalHits = new Set<number>();
		if (phrase.length >= 2) {
			for (let i = 0; i < this.meta.length; i++) {
				if (this.meta[i].title.toLowerCase().includes(phrase)) literalHits.add(i);
			}
		}
		this.lastCorrection = literalHits.size > 0 ? null : this.correct(termSets, phrase);

		const hits: SearchHit[] = [];
		for (let i = 0; i < this.meta.length; i++) {
			const m = this.meta[i];
			const doc = this.docTerms?.[i];
			let score = 0;
			for (const terms of termSets) {
				let setScore = 0;
				for (const q of terms) {
					let best = 0;
					if (doc) {
						for (const t of doc.title) best = Math.max(best, termsMatch(q, t) * 2);
						if (best < 1) for (const t of doc.rest) best = Math.max(best, termsMatch(q, t));
					} else {
						// 분석기 준비 전 폴백: 부분 문자열
						if (m.title.toLowerCase().includes(q)) best = 2;
						else if (
							`${m.tags.join(' ')} ${m.authors.map((a) => a.name).join(' ')}`
								.toLowerCase()
								.includes(q)
						)
							best = 1;
					}
					if (best === 0) {
						setScore = 0;
						break;
					}
					setScore += best;
				}
				// 세트 길이로 정규화해 0~1로: 1.0 = 모든 단어가 제목과 정확 일치
				score = Math.max(score, setScore / (2 * terms.length));
			}
			// 문자 그대로 제목에 든 글은 어떤 추론 매칭보다도 위
			if (literalHits.has(i)) score = Math.max(score, 1.5);
			if (score > 0) hits.push({ ...m, score, source: 'keyword' });
		}
		return hits.sort((a, b) => b.score - a.score).slice(0, k);
	}

	static fuse(
		keyword: SearchHit[],
		semantic: SearchHit[],
		queryTokens: number,
		corrected: boolean,
		k = 20
	): { results: SearchHit[]; led: 'keyword' | 'semantic' } {
		if (keyword.length === 0) return { results: semantic.slice(0, k), led: 'semantic' };
		const kwStrong = keyword[0].score >= (queryTokens <= 3 ? 0.65 : 0.9);
		const kwIds = new Set(keyword.map((h) => h.id));
		const overlap = semantic.slice(0, 10).filter((h) => kwIds.has(h.id)).length;
		const distrustSemantic =
			!corrected && kwStrong && overlap / Math.min(10, keyword.length) < 0.3;

		const [lead, fill] = distrustSemantic ? [keyword, semantic] : [semantic, keyword];
		const seen = new Set<string>();
		const results: SearchHit[] = [];
		for (const h of [...lead, ...fill]) {
			if (seen.has(h.id)) continue;
			seen.add(h.id);
			results.push(h);
			if (results.length >= k) break;
		}
		return { results, led: distrustSemantic ? 'keyword' : 'semantic' };
	}

	private student: StudentAssets | null = null;
	private studentLoading: Promise<boolean> | null = null;
	private lexShards = new Map<number, Promise<LexShard | null>>();

	// false when the index ships no student assets (or they are inconsistent)
	loadStudent(): Promise<boolean> {
		this.studentLoading ??= Promise.all([
			fetch(`${this.base}/student_meta.json`).then(okJson),
			this.loadMeta()
		])
			.then(([m]) =>
				Promise.all([
					m,
					fetch(`${this.base}/student_vocab_i8.bin`).then(okBuf),
					fetch(`${this.base}/student_proj_i8.bin`).then(okBuf),
					fetch(`${this.base}/student_params_f32.bin`).then(okBuf),
					fetch(`${this.base}/student_doc_vectors_i8.bin`).then(okBuf),
					fetch(`${this.base}/student_doc_scales.json`).then(okJson),
					fetch(`${this.base}/student_lex_meta.json`)
						.then(okJson)
						.catch(() => null)
				])
			)
			.then(([m, vocab, proj, params, docs, ds, lexMeta]) => {
				const V: number = m.vocab_size;
				const dim: number = m.dim;
				const out: number = m.out_dim;
				const M: number = m.members;
				const N: number = m.doc_ids.length;
				const tail = (m.version ?? 1) >= 2 ? 2 : 1; // v2 appends ext_weight
				const block = V + 2 * out + tail;
				if (
					m.terms.length !== V ||
					vocab.byteLength !== V * dim ||
					proj.byteLength !== M * out * dim ||
					params.byteLength !== M * block * 4 ||
					docs.byteLength !== N * out ||
					ds.scales.length !== N
				)
					throw new Error('student asset size mismatch');
				const p = new Float32Array(params);
				const metaRow = new Map(this.meta!.map((d, i) => [String(d.id), i]));
				const members = [...Array(M).keys()];
				this.student = {
					index: new Map((m.terms as string[]).map((t, i) => [t, i])),
					dim,
					outDim: out,
					members: M,
					maxSlots: m.max_slots ?? 16,
					fbMinLen: m.fallback?.min_len ?? 3,
					fbMinRatio: m.fallback?.min_ratio ?? 0.5,
					vocab: new Int8Array(vocab),
					proj: new Int8Array(proj),
					termScale: members.map((k) => p.subarray(k * block, k * block + V)),
					projScale: members.map((k) => p.subarray(k * block + V, k * block + V + out)),
					projBias: members.map((k) => p.subarray(k * block + V + out, k * block + V + 2 * out)),
					fbWeight: members.map((k) => p[k * block + V + 2 * out]),
					extWeight: members.map((k) => (tail === 2 ? p[k * block + V + 2 * out + 1] : 0)),
					docs: new Int8Array(docs),
					docScales: new Float32Array(ds.scales),
					docRow: Int32Array.from(m.doc_ids as string[], (id) => metaRow.get(String(id)) ?? -1),
					// lexicon vectors live in the input basis the student was trained on
					lex:
						tail === 2 &&
						lexMeta?.packs &&
						m.basis_sha1 &&
						lexMeta.basis_sha1 === m.basis_sha1 &&
						lexMeta.dim === dim
							? { packs: lexMeta.packs, index: lexMeta.index }
							: null
				};
				return true;
			})
			.catch(() => false);
		return this.studentLoading;
	}

	// One lexicon shard = a byte range of a pack file; a server that ignores Range sends the whole pack
	private lexShard(term: string): Promise<LexShard | null> {
		const s = this.student!;
		const lex = s.lex!;
		const id = fnv1a(term) % lex.index.length;
		let shard = this.lexShards.get(id);
		if (!shard) {
			const [pack, start, length] = lex.index[id];
			shard = fetch(`${this.base}/${lex.packs[pack]}`, {
				headers: { Range: `bytes=${start}-${start + length - 1}` }
			})
				.then(async (r) => {
					const buf = await okBuf(r);
					return r.status === 206 ? buf : buf.slice(start, start + length);
				})
				.then((buf) => {
					const dv = new DataView(buf);
					const count = dv.getUint32(0, true);
					const jsonLen = dv.getUint32(4, true);
					const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, jsonLen)));
					return {
						index: new Map((head.terms as string[]).map((t, i) => [t, i])),
						q: new Int8Array(buf, 8 + jsonLen, count * s.dim),
						scales: new Float32Array(head.scales)
					};
				})
				.catch(() => null);
			this.lexShards.set(id, shard);
		}
		return shard;
	}

	// Longest in-vocabulary substring of an unknown noun, leftmost on ties
	private studentFallback(t: string): number | undefined {
		const { index, fbMinRatio } = this.student!;
		for (let L = t.length - 1; L >= 2 && L / t.length >= fbMinRatio; L--) {
			for (let i = 0; i + L <= t.length; i++) {
				const row = index.get(t.slice(i, i + L));
				if (row !== undefined) return row;
			}
		}
		return undefined;
	}

	// Query vector in the teacher space; null when no token maps to anything.
	// Slot counts (core exact / lexicon exact / substring fallback) feed the relevance gate.
	async studentQuery(text: string): Promise<StudentEncoding | null> {
		const s = this.student;
		if (!s) return null;
		const toks = studentTokens(text);
		const shards = new Map<string, LexShard | null>();
		if (s.lex) {
			const miss = toks.filter(({ t }) => !s.index.has(t)).map(({ t }) => t);
			await Promise.all(miss.map(async (t) => shards.set(t, await this.lexShard(t))));
		}
		type Slot =
			| { key: string; row: number; fb: boolean }
			| { key: string; q: Int8Array; off: number; scale: number };
		const slots: Slot[] = [];
		const used = new Set<string>();
		let core = 0;
		let lexicon = 0;
		let fallback = 0;
		for (const { t, kind } of toks) {
			if (slots.length >= s.maxSlots) break;
			let slot: Slot | null = null;
			const row = s.index.get(t);
			if (row !== undefined) {
				slot = { key: `c${row}`, row, fb: false };
			} else {
				const sh = shards.get(t);
				const lr = sh?.index.get(t);
				if (sh && lr !== undefined) {
					slot = { key: `x${t}`, q: sh.q, off: lr * s.dim, scale: sh.scales[lr] };
				} else if (kind === 'n' && t.length >= s.fbMinLen) {
					const fr = this.studentFallback(t);
					if (fr !== undefined) slot = { key: `c${fr}`, row: fr, fb: true };
				}
			}
			if (!slot || used.has(slot.key)) continue;
			used.add(slot.key);
			slots.push(slot);
			if ('q' in slot) lexicon++;
			else if (slot.fb) fallback++;
			else core++;
		}
		if (slots.length === 0) return null;
		const { dim, outDim } = s;
		const pooled = new Float32Array(dim);
		const member = new Float32Array(outDim);
		const q = new Float32Array(outDim);
		for (let m = 0; m < s.members; m++) {
			pooled.fill(0);
			for (const slot of slots) {
				let src: Int8Array;
				let off: number;
				let w: number;
				if ('q' in slot) {
					src = slot.q;
					off = slot.off;
					w = slot.scale * s.extWeight[m];
				} else {
					src = s.vocab;
					off = slot.row * dim;
					w = s.termScale[m][slot.row] * (slot.fb ? s.fbWeight[m] : 1);
				}
				for (let d = 0; d < dim; d++) pooled[d] += src[off + d] * w;
			}
			const base = m * outDim * dim;
			let norm = 0;
			for (let o = 0; o < outDim; o++) {
				let acc = 0;
				const off = base + o * dim;
				for (let d = 0; d < dim; d++) acc += s.proj[off + d] * pooled[d];
				const v = s.projScale[m][o] * acc + s.projBias[m][o];
				member[o] = v;
				norm += v * v;
			}
			norm = Math.sqrt(norm) || 1;
			for (let o = 0; o < outDim; o++) q[o] += member[o] / norm;
		}
		let norm = 0;
		for (let o = 0; o < outDim; o++) norm += q[o] * q[o];
		norm = Math.sqrt(norm) || 1;
		for (let o = 0; o < outDim; o++) q[o] /= norm;
		return { vector: q, core, lexicon, fallback };
	}

	studentSearch(qvec: Float32Array, k = 20): SearchHit[] {
		const s = this.student;
		const meta = this.meta;
		if (!s || !meta) return [];
		const rows: number[] = [];
		const vals: number[] = [];
		for (let i = 0; i < s.docRow.length; i++) {
			const r = s.docRow[i];
			if (r < 0) continue;
			let dot = 0;
			const off = i * s.outDim;
			for (let o = 0; o < s.outDim; o++) dot += s.docs[off + o] * qvec[o];
			rows.push(r);
			vals.push(dot * s.docScales[i]);
		}
		return vals
			.map((_, j) => j)
			.sort((a, b) => vals[b] - vals[a])
			.slice(0, k)
			.map((j) => ({ ...meta[rows[j]], score: vals[j], source: 'semantic' as const }));
	}
}
