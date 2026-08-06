
import { extractTerms, loadAnalyzer, termsMatch } from './korean';

export const EMBED_DIM = 384; // multilingual-e5-small

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

export class SearchIndex {
	private base: string;
	private meta: DocMeta[] | null = null;
	private vectors: Int8Array | null = null;
	private scales: Float32Array | null = null;
	private metaLoading: Promise<void> | null = null;
	private vectorsLoading: Promise<void> | null = null;

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

	loadVectors(): Promise<void> {
		this.vectorsLoading ??= Promise.all([
			fetch(`${this.base}/doc_vectors_i8.bin`).then((r) => r.arrayBuffer()),
			fetch(`${this.base}/vector_scales.json`).then((r) => r.json())
		]).then(([buf, sc]) => {
			if (sc.dim !== EMBED_DIM) throw new Error(`index dim ${sc.dim} != ${EMBED_DIM}`);
			this.vectors = new Int8Array(buf);
			this.scales = new Float32Array(sc.scales);
		});
		return this.vectorsLoading;
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

	private vocabVecs: {
		index: Map<string, number>;
		data: Int8Array;
		scales: Float32Array;
		dim: number;
	} | null = null;
	private vocabVecsLoading: Promise<boolean> | null = null;

	loadVocabVectors(): Promise<boolean> {
		this.vocabVecsLoading ??= Promise.all([
			fetch(`${this.base}/vocab_meta.json`).then((r) => r.json()),
			fetch(`${this.base}/vocab_vectors_i8.bin`).then((r) => r.arrayBuffer())
		])
			.then(([m, buf]) => {
				this.vocabVecs = {
					index: new Map((m.terms as string[]).map((t, i) => [t, i])),
					data: new Int8Array(buf),
					scales: new Float32Array(m.scales),
					dim: m.dim
				};
				return true;
			})
			.catch(() => false);
		return this.vocabVecsLoading;
	}

	async composeEmbedding(text: string): Promise<Float32Array | null> {
		if (!(await this.loadVocabVectors()) || !this.vocabVecs) return null;
		const { index, data, scales, dim } = this.vocabVecs;
		const terms = [...new Set([...extractTerms(text), text.replace(/\s+/g, '').toLowerCase()])];
		const acc = new Float32Array(dim);
		let found = 0;
		for (const term of terms) {
			const i = index.get(term);
			if (i === undefined) continue;
			const off = i * dim;
			for (let d = 0; d < dim; d++) acc[d] += data[off + d] * scales[i];
			found += 1;
		}
		if (found === 0) return null;
		let norm = 0;
		for (let d = 0; d < dim; d++) norm += acc[d] * acc[d];
		norm = Math.sqrt(norm) || 1;
		for (let d = 0; d < dim; d++) acc[d] /= norm;
		return acc;
	}

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

	semanticSearch(qvec: Float32Array, k = 20): SearchHit[] {
		if (!this.meta || !this.vectors || !this.scales) return [];
		const n = this.scales.length;
		const scores = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			let dot = 0;
			const off = i * EMBED_DIM;
			for (let d = 0; d < EMBED_DIM; d++) dot += this.vectors[off + d] * qvec[d];
			scores[i] = dot * this.scales[i];
		}
		return [...scores.keys()]
			.sort((a, b) => scores[b] - scores[a])
			.slice(0, k)
			.map((i) => ({ ...this.meta![i], score: scores[i], source: 'semantic' as const }));
	}
}
