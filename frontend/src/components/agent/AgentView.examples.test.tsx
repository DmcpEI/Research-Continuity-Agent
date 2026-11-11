import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AgentView } from './AgentView';

vi.mock('../../api/client', () => ({ sendAgent: vi.fn() }));
vi.mock('../../hooks/useSources', () => ({
  useSources: () => ({
    data: [
      {
        id: 'src:pdf/grammar',
        title: 'Grammar-Aligned_Decoding',
        kind: 'paper',
        chunk_count: 4,
        latest_path: null,
        content_sha256: null,
        latest_revision_created_at: null,
      },
    ],
    isSuccess: true,
  }),
}));

describe('Agent example tasks', () => {
  it('are buttons that put a task that works into the box', async () => {
    const user = userEvent.setup();
    render(<AgentView />);
    const example = screen.getByRole('button', {
      name: 'Search the knowledge base for “Grammar-Aligned Decoding” and say in one sentence what it is about.',
    });
    expect(screen.getByRole('button', { name: 'List the files in the workspace folder.' })).toBeInTheDocument();
    expect(screen.queryByText(/rank best precision/)).not.toBeInTheDocument();

    await user.click(example);
    expect(screen.getByRole('textbox', { name: 'Describe a task' })).toHaveValue(
      'Search the knowledge base for “Grammar-Aligned Decoding” and say in one sentence what it is about.',
    );
  });

  it('does not claim the agent checks grounding', () => {
    render(<AgentView />);
    expect(screen.queryByText(/grounded/i)).not.toBeInTheDocument();
  });
});
