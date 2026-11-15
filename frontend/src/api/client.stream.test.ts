import { ApiError, sendChat } from './client';

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

const finalPayload = { answer: 'JamPacker packs boxes.', citations: [], outcome: 'answered', grounded: true, trace: null };

describe('sendChat over the stage stream', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('reports each stage and resolves with the final payload, even when events split across chunks', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      streamOf([
        'event: stage\ndata: {"stage": "retrieving", "elapsed_ms": 3}\n\nevent: sta',
        'ge\ndata: {"stage": "generating", "elapsed_ms": 900}\n\n',
        `event: final\ndata: ${JSON.stringify(finalPayload)}\n\n`,
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);
    const stages: string[] = [];

    const result = await sendChat('What is JamPacker?', 'c1', 'qwen3.5:4b', [], { onStage: (s) => stages.push(s) });

    expect(stages).toEqual(['retrieving', 'generating']);
    expect(result).toEqual(finalPayload);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/chat/stream');
  });

  it('turns an error event into an ApiError with the server detail', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => streamOf(['event: error\ndata: {"detail": "Generation failed: Ollama is down"}\n\n'])));
    await expect(sendChat('q')).rejects.toMatchObject({ detail: 'Generation failed: Ollama is down' });
    await expect(sendChat('q')).rejects.toBeInstanceOf(ApiError);
  });

  it('fails plainly when the stream ends without an answer', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => streamOf(['event: stage\ndata: {"stage": "retrieving", "elapsed_ms": 1}\n\n'])));
    await expect(sendChat('q')).rejects.toThrow('The connection closed before an answer arrived.');
  });
});
