/*
	쿼리 임베딩 워커. multilingual-e5-small(int8 ONNX, 약 118MB)을
	transformers.js로 브라우저 안에서 실행한다 — 서버는 관여하지 않고,
	모델 파일은 HF 허브에서 받아 브라우저 캐시에 남는다(다운로드는 최초 1회).
	메인 스레드가 멈추지 않도록 로딩·추론 모두 워커에서 한다.
*/
import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';

/*
	1순위는 자체 서빙하는 가지치기 모델(37MB — 원본 118MB에서 한국어·영어에
	불필요한 어휘를 제거, 코사인 동등성 1.0000 검증). 임베딩 공간이 원본과
	동일하므로 문서 벡터 인덱스는 그대로 쓴다. 실패하면 HF 허브 원본으로 폴백.
*/
env.allowLocalModels = true;

/* 소비자가 load 메시지로 덮어쓸 수 있는 기본값 */
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

/*
	한 후보(백엔드 x 모델)를 끝까지 검증한다: 세션 생성뿐 아니라 실제
	임베딩 한 번까지. WebGPU는 어댑터가 잡혀도 세션 생성이 실패하거나
	아예 멈추는 환경이 있어(q8 커널 미지원 등) 타임아웃과 웜업이 없으면
	"로딩 중"에 영원히 갇힌다.
*/
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
		/*
			강등 사다리: WebGPU(로컬) → WASM(로컬) → WASM(허브 원본).
			WebGPU가 로컬에서 깨지면 허브에서도 깨지므로 다시 시도하지 않는다.
		*/
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
