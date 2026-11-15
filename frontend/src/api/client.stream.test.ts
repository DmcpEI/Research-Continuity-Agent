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

describe('sendChat stream parsing edges', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('accepts CRLF line endings, as a proxy may rewrite them', async () => {
    const crlf = `event: stage\r\ndata: {"stage": "retrieving", "elapsed_ms": 1}\r\n\r\nevent: final\r\ndata: ${JSON.stringify(finalPayload)}\r\n\r\n`;
    vi.stubGlobal('fetch', vi.fn(async () => streamOf([crlf])));
    const stages: string[] = [];
    await expect(sendChat('q', undefined, undefined, [], { onStage: (s) => stages.push(s) })).resolves.toEqual(finalPayload);
    expect(stages).toEqual(['retrieving']);
  });
});

describe('sendChat token events', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('passes each draft piece on in order, then resolves with the final payload', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamOf([
          'event: stage\ndata: {"stage": "generating", "elapsed_ms": 5}\n\n',
          'event: token\ndata: {"text": "Jam"}\n\nevent: token\ndata: {"text": "Packer\\n\\npacks."}\n\n',
          `event: final\ndata: ${JSON.stringify(finalPayload)}\n\n`,
        ]),
      ),
    );
    const pieces: string[] = [];
    const result = await sendChat('q', undefined, undefined, [], { onToken: (text) => pieces.push(text) });
    expect(pieces).toEqual(['Jam', 'Packer\n\npacks.']);
    expect(result).toEqual(finalPayload);
  });
});

describe('sendChat with a malformed token event', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('skips a token event without text instead of adding "undefined"', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamOf(['event: token\ndata: {}\n\nevent: token\ndata: {"text": "ok"}\n\n', `event: final\ndata: ${JSON.stringify(finalPayload)}\n\n`]),
      ),
    );
    const pieces: string[] = [];
    await sendChat('q', undefined, undefined, [], { onToken: (text) => pieces.push(text) });
    expect(pieces).toEqual(['ok']);
  });
});
