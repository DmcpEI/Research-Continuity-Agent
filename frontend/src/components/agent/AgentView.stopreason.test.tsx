import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { sendAgent } from '../../api/client';
import { AgentView, stopReasonText } from './AgentView';

vi.mock('../../api/client', async () => {
  const actual = await vi.importActual<typeof import('../../api/client')>('../../api/client');
  return { ...actual, sendAgent: vi.fn() };
});
vi.mock('../../hooks/useSources', () => ({ useSources: () => ({ data: [], isSuccess: true }) }));

describe('Agent stop reasons', () => {
  it('turns every reason into a sentence, and keeps an unknown one visible', () => {
    for (const reason of [
      'final_answer',
      'empty_final_answer',
      'unparsed_tool_call',
      'clarification_needed',
      'max_iterations',
      'error',
      'fallback_no_tools',
    ]) {
      expect(stopReasonText(reason)).toMatch(/^[A-Z].*\.$/);
      expect(stopReasonText(reason)).not.toContain('_');
    }
    expect(stopReasonText('something_new')).toBe('Stopped: something_new');
  });

  it('says under the answer why a run did not finish normally', async () => {
    vi.mocked(sendAgent).mockResolvedValueOnce({
      answer: 'Partial findings.',
      trace: {
        query: 'q', iterations: 10, tool_calls: [], model: 'qwen3.5:4b', prompt_tokens: 0,
        completion_tokens: 0, total_latency_ms: 1, stopped_reason: 'max_iterations', warnings: [],
      },
      error: null,
      model: 'qwen3.5:4b',
    } as never);
    const user = userEvent.setup();
    render(<AgentView />);
    await user.type(screen.getByRole('textbox', { name: 'Describe a task' }), 'Do a long thing.{Enter}');
    expect(await screen.findAllByText(stopReasonText('max_iterations'))).not.toHaveLength(0);
  });
});
