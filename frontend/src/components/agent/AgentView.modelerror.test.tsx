import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ApiError, sendAgent } from '../../api/client';
import { AgentView } from './AgentView';

vi.mock('../../api/client', async () => {
  const actual = await vi.importActual<typeof import('../../api/client')>('../../api/client');
  return { ...actual, sendAgent: vi.fn() };
});
vi.mock('../../hooks/useSources', () => ({ useSources: () => ({ data: [], isSuccess: true }) }));

describe('Agent model that cannot call tools', () => {
  it('says so in plain words and opens the model picker', async () => {
    const detail = 'The agent model gemma3:12b cannot call tools. Pick another model in the model picker.';
    vi.mocked(sendAgent).mockRejectedValueOnce(new ApiError(409, detail, `{"detail":"${detail}"}`));
    const opened = vi.fn();
    window.addEventListener('rca-open-model-picker', opened);
    const user = userEvent.setup();
    try {
      render(<AgentView />);
      await user.type(screen.getByRole('textbox', { name: 'Describe a task' }), 'List the files.{Enter}');
      expect((await screen.findAllByText(detail)).length).toBeGreaterThan(0);
      expect(screen.queryByText(/API 409/)).not.toBeInTheDocument();
      expect(opened).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener('rca-open-model-picker', opened);
    }
  });
});
