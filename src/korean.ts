/*
	키워드층(T0)의 한국어 처리 도구.
	· garu(WASM 형태소 분석기, 1.6MB)로 명사만 추려 조사·어미에 흔들리지 않게 하고
	· 자모 단위 편집거리로 한 글자 오타("최저임긍")를 흡수한다.
	garu 로딩에 실패해도 검색은 살아야 하므로, 사용처는 항상 null을 각오한다.
*/
import type { Garu } from 'garu-ko';

let garu: Garu | null = null;
let loading: Promise<Garu | null> | null = null;

export function loadAnalyzer(): Promise<Garu | null> {
	loading ??= import('garu-ko')
		.then((mod) => mod.Garu.load())
		.then((instance) => (garu = instance))
		.catch(() => null); // 실패 시 부분 문자열 매칭으로 폴백
	return loading;
}

/*
	규칙 기반 조사 제거 — 분석기(WASM)가 막힌 환경의 폴백.
	긴 조사부터 시도하고, 남는 몸통이 2자 이상일 때만 벗긴다.
*/
const PARTICLES = [
	'에서부터', '으로부터', '이라도', '에게서', '으로서', '으로써', '까지', '부터', '처럼',
	'보다', '에게', '한테', '께서', '이나', '이란', '라는', '든지', '마다', '조차', '밖에',
	'이랑', '하고', '은', '는', '이', '가', '을', '를', '과', '와', '도', '만', '의', '에',
	'로', '랑', '라', '뿐'
].sort((a, b) => b.length - a.length);

export function stripParticle(word: string): string {
	if (!hasHangul(word)) return word;
	for (const p of PARTICLES) {
		if (word.length - p.length >= 2 && word.endsWith(p)) return word.slice(0, -p.length);
	}
	return word;
}

/* 명사 추출. 분석기가 없으면 공백 분리 + 조사 제거로 물러선다. */
export function extractTerms(text: string): string[] {
	const cleaned = text.trim();
	if (!cleaned) return [];
	if (garu) {
		const nouns = garu.nouns(cleaned, { includeSL: true });
		if (nouns.length > 0) return nouns.map((n) => n.toLowerCase());
	}
	return cleaned.toLowerCase().split(/\s+/).filter(Boolean).map(stripParticle);
}

/* 한글 음절을 자모열로 편다. "각" -> "ㄱㅏㄱ". 한글이 아니면 그대로 둔다. */
const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ';
const JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];

export function toJamo(text: string): string {
	let out = '';
	for (const ch of text) {
		const code = ch.charCodeAt(0) - 0xac00;
		if (code < 0 || code > 11171) {
			out += ch;
			continue;
		}
		out += CHO[Math.floor(code / 588)] + JUNG[Math.floor((code % 588) / 28)] + JONG[code % 28];
	}
	return out;
}

/* 편집거리가 maxDist 이하인지만 판단한다 — 밴드 밖은 바로 포기해 짧게 끝낸다. */
export function withinEditDistance(a: string, b: string, maxDist: number): boolean {
	if (Math.abs(a.length - b.length) > maxDist) return false;
	let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
	for (let i = 1; i <= a.length; i++) {
		const cur = [i];
		let rowMin = i;
		for (let j = 1; j <= b.length; j++) {
			cur[j] = Math.min(
				prev[j] + 1,
				cur[j - 1] + 1,
				prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
			);
			rowMin = Math.min(rowMin, cur[j]);
		}
		if (rowMin > maxDist) return false;
		prev = cur;
	}
	return prev[b.length] <= maxDist;
}

export function hasHangul(text: string): boolean {
	return /[가-힣]/.test(text);
}

function fuzzyEqual(a: string, b: string): boolean {
	if (a.length < 2 || b.length < 2) return false;
	/*
		퍼지 비교는 한글 전용이다. 라틴 문자에 편집거리를 허용하면 짧은 영단어끼리
		죄다 이웃이 된다(tax ↔ tale) — 영어 오타는 교정하지 않고 그대로 두는 게
		낫고, 의미 검색(다국어 임베딩)이 원문을 제대로 처리한다.
	*/
	if (!hasHangul(a) || !hasHangul(b)) return false;
	const budget = Math.min(a.length, b.length) >= 4 ? 2 : 1; // 자모 기준: 한 글자 오타 ≈ 1~2
	return withinEditDistance(toJamo(a), toJamo(b), budget);
}

/*
	두 단어가 "사실상 같은 말"인가.
	같음 > 포함 > 오타 1개 > "조사 달린 오타"(문서 단어 길이만큼 잘라 재비교) 순.
	마지막 단계가 "인플레이숀의" ↔ "인플레이션" 을 잇는다 — 형태소 분석기는
	사전에 없는 오타를 제대로 못 자르므로, 표면형을 직접 견줘야 한다.
*/
export function termsMatch(query: string, doc: string): number {
	if (query === doc) return 1;
	if (Math.min(query.length, doc.length) >= 2) {
		// 쿼리가 문서 단어 안에 통째로 들어가는 건 강한 신호("세금" ⊆ "세금인상")
		if (doc.includes(query)) return 0.9;
		/*
			반대 방향은 문서 단어가 쿼리 대부분을 덮을 때만 —
			결합형 쿼리("인플레이숀의원인")가 짧은 단어("원인")를 부분 포함한다고
			전체가 일치한 것처럼 치면 잡음이 쏟아진다.
		*/
		if (query.includes(doc) && doc.length >= query.length * 0.6) return 0.9;
	}
	if (fuzzyEqual(query, doc)) return 0.7;
	/*
		접두 비교는 짧은 쪽이 긴 쪽의 절반 이상을 덮을 때만.
		"자유시장경제는왜옳은가"의 앞 두 글자가 "자유"와 같다고
		전체를 일치시키면 긴 쿼리가 아무 데나 들러붙는다.
	*/
	/*
		접두 비교는 겹치는 조각이 3자 이상일 때만. 2자 조각의 퍼지 비교는
		"정치시장"의 '정치'가 '정리'(핵심정리)와 자모 1차이로 붙는 식의
		오탐을 양산한다 — 2자짜리 진짜 오타("쎄금")는 전체 퍼지 비교와
		교정 사전이 따로 받아낸다.
	*/
	const ratio = Math.min(query.length, doc.length) / Math.max(query.length, doc.length);
	if (ratio >= 0.5 && Math.min(query.length, doc.length) >= 3) {
		// 조사 달린 쿼리("인플레이숀의") ↔ 문서 단어("인플레이션")
		if (query.length > doc.length && fuzzyEqual(query.slice(0, doc.length), doc)) return 0.65;
		// 오타 쿼리("쎄금") ↔ 문서 복합어("세금인상") — 앞부분끼리 견준다
		if (doc.length > query.length && fuzzyEqual(query, doc.slice(0, query.length))) return 0.65;
	}
	return 0;
}
