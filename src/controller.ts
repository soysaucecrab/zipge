/*
	층위 검색 흐름의 조율자. 쿼리마다 키워드 결과(T0)를 즉시 내보내고,
	모델이 준비되면 같은 쿼리를 시맨틱 결과(T1)로 승격한다.
	응답이 뒤늦게 도착한 옛 쿼리는 시퀀스 번호로 걸러 버린다.
*/
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

	/*
		WASM 힙은 줄어들지 않고 자라기만 하므로, 일정 횟수 검색마다 워커를
		갈아끼워 누적을 리셋한다. 모델이 로컬 37MB라 재기동이 수 초면 끝나서,
		응답을 다 내보낸 직후 백그라운드로 갈아끼우면 사용자는 거의 못 느낀다.
	*/
	private static RECYCLE_AFTER_EMBEDS = 10;

	/*
		모델이 떠 있는 동안 브라우저 메모리를 수백 MB 이상 붙잡는다.
		한동안 검색이 없으면 워커째 내려서 돌려준다 — WASM 힙은 한 번 자라면
		줄지 않으므로 dispose가 아니라 종료여야 실제로 반납된다.
		다음 검색 때 warmup이 새 워커를 만들고 모델은 캐시에서 수 초 만에 돌아온다.
	*/
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
	/* "원래 문구로 검색" 요청(?exact=1)에서는 어떤 교정도 하지 않는다 */
	allowCorrection = true;
	/*
		원문 그대로 모드: 추론(형태소·퍼지·시맨틱)을 전부 끄고 제목에
		문구가 통째로 든 글만 보여준다. 모델도 로딩하지 않는다.
	*/
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

	/* 검색창 포커스 시점에 부른다: 메타 로딩 + 워커/모델/벡터를 미리 데운다. */
	async warmup(): Promise<void> {
		const metaReady = this.index.loadMeta();
		/*
			형태소 기반 T0 준비(1.6MB WASM). 끝나는 대로 현재 쿼리를 명사 매칭으로
			다시 평가한다 — 임베딩 모델(118MB)보다 훨씬 먼저 도착하는 중간 개선이다.
		*/
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
				/*
					분석기가 늦게 도착했지만 교정이 생겼다면(오타·띄어쓰기),
					이미 나간 임베딩 요청은 원문 기준이다 — 교정문으로 다시 청한다.
					새 시퀀스 번호를 받아야 원문 응답이 먼저 와서 자리를 차지해도
					교정문 응답이 그것을 이긴다.
				*/
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
				/* 벡터를 못 받으면 시맨틱만 조용히 비활성 — T0는 살아 있다 */
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
			/*
				모델이 죽었는데(락다운 모드의 WASM 차단 등) 답을 기다리는 검색이
				있으면, 어휘 합성 임베딩으로 같은 검색을 마저 끝낸다.
			*/
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

	/* 모델 없이 어휘 벡터 평균으로 시맨틱을 근사 — 실패하면 조용히 키워드만 남는다 */
	private async composedFallback(id: number, query: string): Promise<void> {
		const vector = await this.index.composeEmbedding(this.lastCorrection ?? query);
		if (!vector || id !== this.seq) return;
		this.emitSemantic(query, vector);
	}

	/* 디바운스는 호출자 몫. 호출마다 T0가 즉시, T1이 준비되는 대로 나간다. */
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
