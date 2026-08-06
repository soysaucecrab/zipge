import { SearchIndex } from './core';
import type { SearchHit } from './core';
import type { EmbedWorkerConfig, WorkerIn, WorkerOut } from './embed-worker';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';
export type SearchEmit = {
	query: string;
	results: SearchHit[];
	tier: 'keyword' | 'semantic';
	correctedTo?: string;
	tags?: { name: string; count: number }[];
};

export class SearchController {
	readonly index: SearchIndex;
	private worker: Worker | null = null;
	private seq = 0;
	private pending = new Map<number, string>();
	private lastKeyword: SearchHit[] = [];
	private lastCorrection: string | null = null;
	private idleTimer: ReturnType<typeof setTimeout> | null = null;
	private embedCount = 0;

	private static RECYCLE_AFTER_EMBEDS = 10;

	private static IDLE_UNLOAD_MS = 3 * 60_000;

	private armIdleUnload(): void {
		if (this.idleTimer) clearTimeout(this.idleTimer);
		this.idleTimer = setTimeout(() => {
			// 응답을 기다리는 중이면(모델 다운로드 포함) 죽이지 않고 미룬다
			if (this.pending.size > 0) {
				this.armIdleUnload();
				return;
			}
			this.worker?.terminate();
			this.worker = null;
			this.status = 'idle';
			this.onStatus('idle');
		}, SearchController.IDLE_UNLOAD_MS);
	}

	status: ModelStatus = 'idle';
	onResults: (emit: SearchEmit) => void = () => {};
	onStatus: (status: ModelStatus, error?: string) => void = () => {};
	allowCorrection = true;
	exactMode = false;

	private workerConfig?: EmbedWorkerConfig;

	constructor(options?: {
		indexBase?: string;
		worker?: EmbedWorkerConfig;
		recycleAfterEmbeds?: number;
		idleUnloadMs?: number;
	}) {
		this.index = new SearchIndex(options?.indexBase);
		this.workerConfig = options?.worker;
		if (options?.recycleAfterEmbeds) SearchController.RECYCLE_AFTER_EMBEDS = options.recycleAfterEmbeds;
		if (options?.idleUnloadMs) SearchController.IDLE_UNLOAD_MS = options.idleUnloadMs;
	}

	async warmup(): Promise<void> {
		const metaReady = this.index.loadMeta();
		this.index.prepareKeyword().then(() => {
			const current = this.pending.get(this.seq);
			if (current !== undefined) {
				this.lastKeyword = this.index.keywordSearch(current);
				this.lastCorrection = this.allowCorrection ? this.index.lastCorrection : null;
				this.onResults({
					query: current,
					results: this.lastKeyword,
					tier: 'keyword',
					correctedTo: this.lastCorrection ?? undefined,
					tags: this.matchedTags(current)
				});
				if (this.lastCorrection && this.status !== 'error' && this.worker) {
					const id = ++this.seq;
					this.pending.set(id, current);
					this.worker.postMessage({
						type: 'embed',
						id,
						text: this.lastCorrection
					} satisfies WorkerIn);
					this.armIdleUnload();
				} else if (this.lastCorrection && this.status === 'error') {
					this.composedFallback(++this.seq, current);
				}
			}
		});
		if (!this.worker) {
			this.spawnWorker();
			this.index.loadVectors().catch(() => {
			});
		}
		await metaReady;
	}

	private spawnWorker(): void {
		this.worker = new Worker(new URL('./embed-worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (ev: MessageEvent<WorkerOut>) => this.handle(ev.data);
		this.worker.postMessage({ type: 'load', config: this.workerConfig } satisfies WorkerIn);
		this.embedCount = 0;
	}

	private maybeRecycle(): void {
		if (this.embedCount < SearchController.RECYCLE_AFTER_EMBEDS) return;
		if (this.pending.size > 0 || !this.worker) return;
		this.worker.terminate();
		this.spawnWorker(); // 즉시 예열 — 누적 힙만 버리고 준비 상태는 유지
	}

	private matchedTags(query: string): { name: string; count: number }[] {
		const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
		if (this.lastCorrection)
			terms.push(...this.lastCorrection.toLowerCase().split(/\s+/).filter(Boolean));
		return this.index.matchTags(terms);
	}

	private handle(msg: WorkerOut): void {
		if (msg.type === 'status') {
			this.status = msg.status === 'unloaded' ? 'idle' : msg.status;
			this.onStatus(this.status, msg.error);
			if (this.status === 'error') {
				const current = this.pending.get(this.seq);
				if (current !== undefined) {
					this.pending.delete(this.seq);
					this.composedFallback(this.seq, current);
				}
			}
			return;
		}
		const query = this.pending.get(msg.id);
		this.pending.delete(msg.id);
		this.embedCount += 1;
		this.armIdleUnload(); // 유휴 카운트는 마지막 응답 시점부터
		this.maybeRecycle();
		if (query === undefined || msg.id !== this.seq) return; // 오래된 응답
		this.emitSemantic(query, msg.vector);
	}

	private emitSemantic(query: string, vector: Float32Array): void {
		this.index.loadVectors().then(() => {
			const tokens = query.trim().split(/\s+/).filter(Boolean).length;
			const fused = SearchIndex.fuse(
				this.lastKeyword,
				this.index.semanticSearch(vector),
				tokens,
				this.lastCorrection !== null
			);
			this.onResults({
				query,
				results: fused.results,
				tier: fused.led,
				correctedTo: this.lastCorrection ?? undefined,
				tags: this.matchedTags(query)
			});
		});
	}

	private async composedFallback(id: number, query: string): Promise<void> {
		const vector = await this.index.composeEmbedding(this.lastCorrection ?? query);
		if (!vector || id !== this.seq) return;
		this.emitSemantic(query, vector);
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
		if (this.status !== 'error' && this.worker) {
			this.pending.set(id, query);
			// 오타·띄어쓰기가 교정됐다면 임베딩도 교정문으로 — 의미 검색까지 살린다
			this.worker.postMessage({
				type: 'embed',
				id,
				text: this.lastCorrection ?? query
			} satisfies WorkerIn);
			this.armIdleUnload();
		} else if (this.status === 'error') {
			// 모델이 이미 죽어 있는 세션: 곧바로 합성 임베딩 경로로
			this.composedFallback(id, query);
		}
	}

	destroy(): void {
		if (this.idleTimer) clearTimeout(this.idleTimer);
		this.worker?.terminate();
		this.worker = null;
	}
}
