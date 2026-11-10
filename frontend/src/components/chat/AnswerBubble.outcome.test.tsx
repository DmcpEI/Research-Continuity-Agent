import { render, screen } from '@testing-library/react';
import { AnswerBubble } from './AnswerBubble';

type BubbleProps = Parameters<typeof AnswerBubble>[0];

function renderBubble(props: Partial<BubbleProps>) {
  return render(
    <AnswerBubble content="Some reply" citations={[]} grounded={false} trace={null} {...props} />,
  );
}

describe('Answer outcome badge', () => {
  it('shows a cited answer as grounded', () => {
    renderBubble({ outcome: 'answered', grounded: true });
    expect(screen.getByText('✓ Grounded')).toBeInTheDocument();
  });

  it('shows an uncited answer as unverified', () => {
    renderBubble({ outcome: 'unverified' });
    expect(screen.getByText('⚠ Unverified')).toBeInTheDocument();
  });

  it('shows a refusal as "no evidence found", not as an unverified answer', () => {
    renderBubble({ outcome: 'not_found' });
    expect(screen.getByText('No evidence found')).toBeInTheDocument();
    expect(screen.queryByText('⚠ Unverified')).not.toBeInTheDocument();
  });

  it('shows no badge on a reply about the chat itself', () => {
    const { container } = renderBubble({ outcome: 'conversational' });
    expect(container.querySelector('.grounded-badge')).toBeNull();
    expect(screen.getByText('Some reply')).toBeInTheDocument();
  });

  it('shows a failed request as failed, not as an unverified answer', () => {
    renderBubble({ outcome: 'error' });
    expect(screen.getByText('Request failed')).toBeInTheDocument();
    expect(screen.queryByText('⚠ Unverified')).not.toBeInTheDocument();
  });

  it.each([
    [true, '✓ Grounded'],
    [false, '⚠ Unverified'],
  ])('falls back to the grounded flag (%s) for messages saved without an outcome', (grounded, label) => {
    renderBubble({ grounded });
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('falls back to the grounded flag for an outcome this build does not know', () => {
    renderBubble({ outcome: 'toString' as BubbleProps['outcome'], grounded: true });
    expect(screen.getByText('✓ Grounded')).toBeInTheDocument();
  });
});
