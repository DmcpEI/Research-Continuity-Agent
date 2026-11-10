import { type AnswerOutcome, type ApiCitation } from '../../api/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CitationChip } from './CitationChip';
import { QueryTrace } from './QueryTrace';

type BubbleOutcome = AnswerOutcome | 'error';

const BADGES: Record<BubbleOutcome, { label: string; className: string } | null> = {
  answered: { label: '✓ Grounded', className: 'is-grounded' },
  unverified: { label: '⚠ Unverified', className: 'is-unverified' },
  not_found: { label: 'No evidence found', className: 'is-not-found' },
  conversational: null,
  error: { label: 'Request failed', className: 'is-error' },
};

type AnswerBubbleProps = {
  content: string;
  citations: ApiCitation[];
  // Messages saved before outcomes existed only have `grounded`.
  outcome?: BubbleOutcome;
  grounded: boolean;
  trace: Record<string, unknown> | null;
  onCitationClick?: (sourceId: string) => void;
};

export function AnswerBubble({
  content,
  citations,
  outcome,
  grounded,
  trace,
  onCitationClick,
}: AnswerBubbleProps) {
  const normalizedCitations = citations.filter((citation, index, all) => {
    const sourceId = citation.source_id?.trim();
    const isValid = sourceId.startsWith('src:') || sourceId.startsWith('chk:');
    if (!isValid) {
      return false;
    }
    return all.findIndex((item) => item.source_id === citation.source_id) === index;
  });

  const displayContent = content
    .replace(/\[\[[\w:/.-]+\]\]/g, '')
    .replace(/\[(?:src|chk):[^\]\s]+\]/g, '')
    .split('\n')
    .map((line) => line.trimEnd())
    .join('\n')
    .trim();

  // An outcome this build does not know (newer server) falls back to the grounded flag too.
  const fallback = grounded ? BADGES.answered : BADGES.unverified;
  const known = outcome !== undefined && Object.prototype.hasOwnProperty.call(BADGES, outcome);
  const badge = known ? BADGES[outcome] : fallback;

  return (
    <article className="answer-bubble">
      {badge ? (
        <header>
          <span className={`grounded-badge ${badge.className}`}>{badge.label}</span>
        </header>
      ) : null}

      <div className="answer-prose">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
      </div>

      {normalizedCitations.length > 0 ? (
        <div className="citations-row">
          {normalizedCitations.map((citation) => (
            <CitationChip
              key={`${citation.source_id}-${citation.title}`}
              onClick={onCitationClick}
              sourceId={citation.source_id}
            />
          ))}
        </div>
      ) : null}

      <QueryTrace trace={trace} />
    </article>
  );
}
