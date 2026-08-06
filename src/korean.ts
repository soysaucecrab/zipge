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

export function extractTerms(text: string): string[] {
	const cleaned = text.trim();
	if (!cleaned) return [];
	if (garu) {
		const nouns = garu.nouns(cleaned, { includeSL: true });
		if (nouns.length > 0) return nouns.map((n) => n.toLowerCase());
	}
	return cleaned.toLowerCase().split(/\s+/).filter(Boolean).map(stripParticle);
}

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
	if (!hasHangul(a) || !hasHangul(b)) return false;
	const budget = Math.min(a.length, b.length) >= 4 ? 2 : 1; // 자모 기준: 한 글자 오타 ≈ 1~2
	return withinEditDistance(toJamo(a), toJamo(b), budget);
}

export function termsMatch(query: string, doc: string): number {
	if (query === doc) return 1;
	if (Math.min(query.length, doc.length) >= 2) {
		// 쿼리가 문서 단어 안에 통째로 들어가는 건 강한 신호("세금" ⊆ "세금인상")
		if (doc.includes(query)) return 0.9;
		if (query.includes(doc) && doc.length >= query.length * 0.6) return 0.9;
	}
	if (fuzzyEqual(query, doc)) return 0.7;
	const ratio = Math.min(query.length, doc.length) / Math.max(query.length, doc.length);
	if (ratio >= 0.5 && Math.min(query.length, doc.length) >= 3) {
		// 조사 달린 쿼리("인플레이숀의") ↔ 문서 단어("인플레이션")
		if (query.length > doc.length && fuzzyEqual(query.slice(0, doc.length), doc)) return 0.65;
		// 오타 쿼리("쎄금") ↔ 문서 복합어("세금인상") — 앞부분끼리 견준다
		if (doc.length > query.length && fuzzyEqual(query, doc.slice(0, query.length))) return 0.65;
	}
	return 0;
}
