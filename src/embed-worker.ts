import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';

env.allowLocalModels = true;

let CONFIG = {
	localModelPath: '/models',
	localModel: 'e5-small-ko',
	hubModel: 'Xenova/multilingual-e5-small'
};

export type EmbedWorkerConfig = Partial<typeof CONFIG>;

export type WorkerIn =
	| { type: 'load'; config?: EmbedWorkerConfig }
	| { type: 'embed'; id: number; text: string }
	| { type: 'unload' };
export type WorkerOut =
	| { type: 'status'; status: 'loading' | 'ready' | 'error' | 'unloaded'; error?: string }
	| { type: 'embedding'; id: number; vector: Float32Array };

let extractor: FeatureExtractionPipeline | null = null;
let loading: Promise<void> | null = null;

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
	return Promise.race([
		p,
		new Promise<never>((_, reject) =>
			setTimeout(() => reject(new Error(`${label} timeout`)), ms)
		)
	]);
}

async function tryLoad(
	model: string,
	device: 'webgpu' | 'wasm',
	ms: number
): Promise<FeatureExtractionPipeline> {
	const opts = device === 'webgpu' ? ({ device: 'webgpu', dtype: 'q8' } as const) : ({ dtype: 'q8' } as const);
	const ex = await withTimeout(pipeline('feature-extraction', model, opts), ms, `${device} load`);
	await withTimeout(
		ex('query: 워밍업', { pooling: 'mean', normalize: true }),
		10_000,
		`${device} warmup`
	);
	return ex;
}

function load(): Promise<void> {
	loading ??= (async () => {
		postMessage({ type: 'status', status: 'loading' } satisfies WorkerOut);
		let hasAdapter = false;
		try {
			const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
			hasAdapter = Boolean(gpu && (await gpu.requestAdapter()));
		} catch {
			hasAdapter = false;
		}
		const ladder: Array<[string, 'webgpu' | 'wasm', number]> = [];
		if (hasAdapter) ladder.push([CONFIG.localModel, 'webgpu', 20_000]);
		ladder.push([CONFIG.localModel, 'wasm', 60_000], [CONFIG.hubModel, 'wasm', 300_000]);

		let lastError = '';
		for (const [model, device, ms] of ladder) {
			try {
				extractor = await tryLoad(model, device, ms);
				postMessage({ type: 'status', status: 'ready' } satisfies WorkerOut);
				return;
			} catch (e) {
				lastError = `${device}/${model}: ${String(e).slice(0, 120)}`;
			}
		}
		loading = null; // 다음 검색에서 재시도 여지를 남긴다
		postMessage({ type: 'status', status: 'error', error: lastError } satisfies WorkerOut);
	})();
	return loading;
}

async function unload(): Promise<void> {
	if (loading) await loading.catch(() => {});
	await extractor?.dispose();
	extractor = null;
	loading = null;
	postMessage({ type: 'status', status: 'unloaded' } satisfies WorkerOut);
}

self.onmessage = async (ev: MessageEvent<WorkerIn>) => {
	const msg = ev.data;
	try {
		if (msg.type === 'load') {
			if (msg.config) CONFIG = { ...CONFIG, ...msg.config };
			env.localModelPath = CONFIG.localModelPath;
			await load();
		}
		if (msg.type === 'unload') await unload();
		if (msg.type === 'embed') {
			await load();
			// e5 계열은 쿼리에 "query: " 접두사를 붙여야 문서("passage: ")와 같은 공간에 놓인다
			const out = await extractor!(`query: ${msg.text}`, { pooling: 'mean', normalize: true });
			const vector = new Float32Array(out.data as Float32Array);
			postMessage({ type: 'embedding', id: msg.id, vector } satisfies WorkerOut, {
				transfer: [vector.buffer]
			});
		}
	} catch (e) {
		postMessage({ type: 'status', status: 'error', error: String(e) } satisfies WorkerOut);
	}
};
