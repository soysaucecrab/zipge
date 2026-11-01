import { SearchIndex } from './core';
import type { SearchHit } from './core';

export type EncoderStatus = 'idle' | 'loading' | 'ready' | 'error';
export type StudentGate = { exact: number; fallbackOnly: number };
export type SearchEmit = {
	query: string;
	results: SearchHit[];
	tier: 'keyword' | 'semantic';
	correctedTo?: string;
	tags?: { name: string; count: number }[];
};

export class SearchController {
	readonly index: SearchIndex;
	private seq = 0;
	private lastKeyword: SearchHit[] = [];
	private lastCorrection: string | null = null;
	private encoderLoading: Promise<boolean> | null = null;

	status: EncoderStatus = 'idle';
	onResults: (emit: SearchEmit) => void = () => {};
	onStatus: (status: EncoderStatus, error?: string) => void = () => {};
	allowCorrection = true;
	exactMode = false;

	/*
		Relevance gate on the top student cosine, keyed on slot composition: a query with an
		exact vocabulary/lexicon hit passes at `exact`, a substring-fallback-only query needs
		`fallbackOnly`. Defaults come from one corpus — recalibrate with pipeline/student/calib_gate.mjs.
	*/
	studentGate: StudentGate = { exact: 0.3, fallbackOnly: 0.635 };

	constructor(options?: { indexBase?: string; studentGate?: Partial<StudentGate> }) {
		this.index = new SearchIndex(options?.indexBase);
		this.studentGate = { ...this.studentGate, ...options?.studentGate };
	}

	async warmup(): Promise<void> {
		const metaReady = this.index.loadMeta();
		void this.index.prepareKeyword();
		void this.loadEncoder();
		await metaReady;
	}

	// Without student assets the controller stays on the keyword tiers
	private loadEncoder(): Promise<boolean> {
		this.encoderLoading ??= (async () => {
			this.setStatus('loading');
			const ok = await this.index.loadStudent();
			if (ok) this.setStatus('ready');
			else this.setStatus('error', 'student assets not found');
			return ok;
		})();
		return this.encoderLoading;
	}

	private setStatus(status: EncoderStatus, error?: string): void {
		this.status = status;
		this.onStatus(status, error);
	}

	private matchedTags(query: string): { name: string; count: number }[] {
		const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
		if (this.lastCorrection)
			terms.push(...this.lastCorrection.toLowerCase().split(/\s+/).filter(Boolean));
		return this.index.matchTags(terms);
	}

	private async semantic(id: number, query: string): Promise<void> {
		// analyzer-backed keyword hits and correction first; the query vector uses the corrected text
		const [ready] = await Promise.all([this.loadEncoder(), this.index.prepareKeyword()]);
		if (id !== this.seq) return;
		this.lastKeyword = this.index.keywordSearch(query);
		this.lastCorrection = this.allowCorrection ? this.index.lastCorrection : null;
		// may wait on one extension-lexicon shard per out-of-vocabulary word
		const enc = ready ? await this.index.studentQuery(this.lastCorrection ?? query) : null;
		if (id !== this.seq) return;
		const hits = enc ? this.index.studentSearch(enc.vector) : [];
		const floor =
			enc && enc.core + enc.lexicon > 0 ? this.studentGate.exact : this.studentGate.fallbackOnly;
		const reached = hits.length > 0 && hits[0].score >= floor;
		const fused = SearchIndex.fuse(
			this.lastKeyword,
			reached ? hits : [],
			query.trim().split(/\s+/).filter(Boolean).length,
			this.lastCorrection !== null
		);
		this.onResults({
			query,
			results: fused.results,
			tier: reached ? fused.led : 'keyword',
			correctedTo: this.lastCorrection ?? undefined,
			tags: this.matchedTags(query)
		});
	}

	async search(query: string): Promise<void> {
		await this.index.loadMeta();
		const id = ++this.seq;
		if (this.exactMode) {
			const literal = this.index.keywordSearch(query).filter((h) => h.score >= 1.5);
			this.onResults({ query, results: literal, tier: 'keyword' });
			return;
		}
		this.lastKeyword = this.index.keywordSearch(query);
		this.lastCorrection = this.allowCorrection ? this.index.lastCorrection : null;
		this.onResults({
			query,
			results: this.lastKeyword,
			tier: 'keyword',
			correctedTo: this.lastCorrection ?? undefined,
			tags: this.matchedTags(query)
		});
		if (!query.trim()) return;
		await this.warmup();
		await this.semantic(id, query);
	}

	// Drops results still in flight; the loaded index stays cached on the instance
	destroy(): void {
		this.seq++;
	}
}
