import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AnswerBubble } from './AnswerBubble';

describe('Citation rendering conformance', () => {
  it('CIT-001: renders valid source chips and supports click', async () => {
    const onCitationClick = vi.fn();
    const user = userEvent.setup();

    render(
      <AnswerBubble
        content="Grounded answer"
        citations={[{ source_id: 'src:pdf/paper_a', title: 'paper_a', excerpt: 'x' }]}
        grounded={true}
        trace={null}
        onCitationClick={onCitationClick}
      />,
    );

    const chip = screen.getByRole('button', { name: 'src:pdf/paper_a' });
    expect(chip).toBeInTheDocument();

    await user.click(chip);
    expect(onCitationClick).toHaveBeenCalledWith('src:pdf/paper_a');
  });

  it('CIT-002: deduplicates repeated source IDs', () => {
    render(
      <AnswerBubble
        content="Answer"
        citations={[
          { source_id: 'src:pdf/paper_a', title: 'a', excerpt: 'x' },
          { source_id: 'src:pdf/paper_a', title: 'a-dup', excerpt: 'y' },
          { source_id: 'src:pdf/paper_b', title: 'b', excerpt: 'z' },
        ]}
        grounded={true}
        trace={null}
      />,
    );

    expect(screen.getAllByRole('button')).toHaveLength(2);
  });

  it('CIT-004: removes inline [[...]] markers while preserving chips', () => {
    render(
      <AnswerBubble
        content={'Result [[src:pdf/paper_a]] with details [[chk:pdf/paper_a:0009]]'}
        citations={[{ source_id: 'src:pdf/paper_a', title: 'a', excerpt: 'x' }]}
        grounded={true}
        trace={null}
      />,
    );

    expect(screen.getByText('Result with details')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'src:pdf/paper_a' })).toBeInTheDocument();
  });

  it('CIT-005: ignores malformed IDs safely and keeps valid entries', () => {
    render(
      <AnswerBubble
        content="Answer"
        citations={[
          { source_id: '', title: 'empty', excerpt: 'x' },
          { source_id: 'paper_a', title: 'bad-prefix', excerpt: 'y' },
          { source_id: 'src:pdf/paper_a', title: 'good', excerpt: 'z' },
        ]}
        grounded={true}
        trace={null}
      />,
    );

    expect(screen.getAllByRole('button')).toHaveLength(1);
  });

  it('CIT-009: citation chip is keyboard operable', async () => {
    const onCitationClick = vi.fn();
    const user = userEvent.setup();

    render(
      <AnswerBubble
        content="Answer"
        citations={[{ source_id: 'src:pdf/paper_a', title: 'a', excerpt: 'x' }]}
        grounded={true}
        trace={null}
        onCitationClick={onCitationClick}
      />,
    );

    await user.tab();
    expect(screen.getByRole('button', { name: 'src:pdf/paper_a' })).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(onCitationClick).toHaveBeenCalledWith('src:pdf/paper_a');
  });
});
