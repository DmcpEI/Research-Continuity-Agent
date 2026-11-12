import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { sendAgent } from '../../api/client';
import { AgentView } from './AgentView';

vi.mock('../../api/client', async () => {
  const actual = await vi.importActual<typeof import('../../api/client')>('../../api/client');
  return { ...actual, sendAgent: vi.fn() };
});
vi.mock('../../hooks/useSources', () => ({ useSources: () => ({ data: [], isSuccess: true }) }));

const reply = (answer: string) =>
  ({
    answer,
    trace: {
      query: 'q', iterations: 1, tool_calls: [], model: 'qwen3.5:4b', prompt_tokens: 0,
      completion_tokens: 0, total_latency_ms: 1, stopped_reason: 'final_answer', warnings: [],
    },
    error: null,
    model: 'qwen3.5:4b',
  }) as never;

describe('Agent tasks are single-shot', () => {
  it('says so, and sends only the task: no history the server would ignore', async () => {
    vi.mocked(sendAgent).mockResolvedValueOnce(reply('First.')).mockResolvedValueOnce(reply('Second.'));
    const user = userEvent.setup();
    render(<AgentView conversationId="agent-main" />);
    expect(screen.getByText(/Each task runs on its own/)).toBeInTheDocument();

    const box = screen.getByRole('textbox', { name: 'Describe a task' });
    await user.type(box, 'List the files.{Enter}');
    await screen.findByText('First.');
    await user.type(box, 'Now read the first one.{Enter}');
    await screen.findByText('Second.');

    expect(vi.mocked(sendAgent).mock.calls).toEqual([['List the files.'], ['Now read the first one.']]);
  });
});
