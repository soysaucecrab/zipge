/*
	시맨틱 검색의 데이터 층. 오프라인에서 미리 계산한 정적 인덱스
	(static/search-index/*)를 읽어 키워드 검색(T0)과 코사인 검색(T1)을 제공한다.
	문서 벡터는 int8 양자화(벡터당 스케일)라 785편 전체가 294KB에 들어간다.
*/

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

	/* 쿼리와 닿는 태그들 — 칩으로 노출해 태그 필터로 가는 길을 연다. */
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

	/*
		T0: 제목·태그·저자 매칭. 형태소 분석기가 준비되면 명사 단위로 비교해
		조사·어미에 흔들리지 않고, 자모 편집거리로 한 글자 오타를 흡수한다.
		분석기 로딩 전·실패 시에는 부분 문자열 일치로 폴백한다.
	*/
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
			/*
				분석기(WASM)가 막힌 환경(락다운 모드 등): 파이프라인이 미리
				토큰화해 둔 사전(terms.json)을 받는다. 쿼리 쪽 명사 추출은
				규칙 기반 조사 제거(stripParticle)가 대신한다.
			*/
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
		/*
			오타 교정 사전: 코퍼스에 실제로 존재하는 단어들.
			분석기가 복합어를 쪼개 놓으므로("사유재산" -> 사유+재산) 인접 결합형도
			함께 넣는다 — 그래야 "사유재솬" 같은 복합어 오타가 반쪽("자유")이 아니라
			온전한 말("사유재산")로 교정된다.
		*/
		const single = this.docTerms.flatMap((d) => [...d.title, ...d.rest]);
		const bigrams = this.docTerms.flatMap((d) =>
			d.title.slice(0, -1).map((t, i) => t + d.title[i + 1])
		);
		this.vocab = [...new Set([...single, ...bigrams])];
	}

	/*
		쿼리를 코퍼스 어휘로 교정한다. 해석(명사/표면형/결합형)마다 각 단어를
		가장 가까운 실제 단어로 바꿔 보고, 전 단어가 이어지는 해석 중 매칭 품질이
		가장 높은 것을 고른다. 품질이 같으면 단어가 긴(더 구체적인) 쪽 —
		"세 금"에서 [세, 금]보다 "세금"이 이긴다.
	*/
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
			/*
				동점이면 원문 글자를 많이 보존한 교정이 이긴다 —
				"미제스 코리아 특별카럼"에서 결합형이 "미제스코리아"(포함 일치)로
				단어를 통째로 버리는 것보다 "특별카럼"->"특별칼럼" 치환이 낫다.
				그다음 동점은 토큰이 적은 쪽("세 금"의 [세,금]보다 "세금" 합침).
			*/
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
		/*
			교정이라 부를 수 있는 건 두 경우뿐이다:
			· 실제 오타 치환이 일어났거나("사유재솬" -> "사유재산")
			· 띄어 친 말이 붙었거나("세 금" -> "세금": 글자 같고 토큰이 줄어듦)
			조사만 떨어져 나간 것("자유와 반공을 위해" -> "자유 반공")은 명사
			추출의 부산물이지 사용자의 실수가 아니다 — 원문이 임베딩에 더 낫다.
		*/
		const mergedSpacing =
			sameChars && best.terms.length < original.split(/\s+/).filter(Boolean).length;
		if (!best.replaced && !mergedSpacing) return null;
		return correctedQuery;
	}

	/*
		키워드층이 알아낸 "사용자가 치려던 말". 오타·띄어쓰기를 문서 어휘로
		교정한 형태로, 시맨틱층의 임베딩 입력으로 쓴다 — "세 금"을 그대로
		임베딩하면 금(gold)이 되지만, 교정된 "세금"은 제대로 세금이 된다.
	*/
	lastCorrection: string | null = null;

	/*
		합성 임베딩 — 모델(WASM/WebGPU)이 막힌 환경의 간이 시맨틱.
		파이프라인이 어휘 단어들을 e5로 미리 임베딩해 둔 표에서 쿼리 단어의
		벡터를 찾아 평균한다. 문서 벡터와 같은 공간이므로 나머지 경로
		(검색·융합·게이트)를 그대로 쓴다. 표는 필요해질 때만 받는다(~1.8MB).
	*/
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
		/*
			쿼리 해석은 두 갈래로 다 해 본다: 형태소 명사와 공백 토큰.
			분석기는 "인플레이션의 원인"을 잘 자르지만, 오타("인플레이숀")는
			사전에 없어 엉뚱하게 쪼갠다 — 그때는 표면형 토큰이 살린다.
		*/
		/*
			커버리지 가드는 "한 단어" 쿼리에서만. "쎄금"처럼 미등록어가 잘려
			['금']만 남으면 반쪽 정확 일치가 오타 매칭을 이겨버리므로 버리지만,
			여러 단어 쿼리("자유와 반공을 위해")에서 조사·기능어가 빠지는 건
			명사 추출의 정상 동작이다 — 여기서 버리면 표면형 해석만 남아
			'위해' 같은 기능어가 사전의 엉뚱한 말('이해')로 오교정된다.
		*/
		const rawTerms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
		let nounTerms = extractTerms(query);
		const queryChars = query.replace(/\s+/g, '').length;
		if (rawTerms.length === 1 && nounTerms.join('').length < queryChars * 0.7) nounTerms = [];
		/*
			띄어 친 한 단어("세 금")는 토큰별로는 너무 짧아 못 잡는다 —
			공백을 지운 결합형을 세 번째 해석으로 함께 시도한다.
		*/
		const joined = query.trim().toLowerCase().replace(/\s+/g, '');
		const termSets = [nounTerms, rawTerms].filter((s) => s.length > 0);
		if (rawTerms.length > 1 && joined.length >= 2) termSets.push([joined]);
		if (termSets.length === 0) return [];

		/*
			1순위는 추론 없는 문자 그대로 일치다. 쿼리가 제목에 통째로 들어
			있으면 그 글이 무조건 위로 오고, 이때는 교정도 발동하지 않는다 —
			멀쩡한 입력을 시스템이 멋대로 고쳐 읽는 것("위해"->"이해")보다
			나쁜 경험은 없다. 교정은 문자 그대로도, 형태소로도 아무것도
			못 찾을 때를 위한 최후수단이다.
		*/
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

	/*
		키워드·시맨틱 결과의 융합. 시맨틱이 항상 이기게 두면 오타·붙여쓰기 쿼리에서
		임베딩 노이즈("쎄금" -> 엉뚱한 글)가 정확한 키워드 결과를 덮어쓴다.
		판정: 키워드 증거가 강한데(제목 거의 일치) 시맨틱 상위와 겹치는 글이
		거의 없으면 시맨틱을 불신하고 키워드를 앞세운다. 그 외엔 시맨틱 주도.
	*/
	static fuse(
		keyword: SearchHit[],
		semantic: SearchHit[],
		queryTokens: number,
		corrected: boolean,
		k = 20
	): { results: SearchHit[]; led: 'keyword' | 'semantic' } {
		if (keyword.length === 0) return { results: semantic.slice(0, k), led: 'semantic' };
		/*
			키워드 주도는 "단어를 찾는" 짧은 쿼리에서만. 긴 자연어 질문은
			키워드 점수가 우연히 높아도(퍼지 잡음 누적) 의미 검색의 영역이다.
			교정문으로 임베딩한 경우엔 시맨틱을 불신할 이유가 사라진다 —
			노이즈의 근원이었던 오타가 이미 걷혔다.
		*/
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

	/* T1: 정규화된 쿼리 벡터와의 코사인 top-k. 785×384 int8 내적은 1ms 미만이다. */
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
