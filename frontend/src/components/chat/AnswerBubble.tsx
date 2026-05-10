import { type ApiCitation } from '../../api/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CitationChip } from './CitationChip';
import { QueryTrace } from './QueryTrace';

type AnswerBubbleProps = {
  content: string;
  citations: ApiCitation[];
  grounded: boolean;
  trace: Record<string, unknown> | null;
  onCitationClick?: (sourceId: string) => void;
};

export function AnswerBubble({
  content,
  citations,
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
    .split('\n')
    .map((line) => line.trimEnd())
    .join('\n')
    .trim();

  return (
    <article className="answer-bubble">
      <header>
        <span className={`grounded-badge ${grounded ? 'is-grounded' : 'is-unverified'}`}>
          {grounded ? '✓ Grounded' : '⚠ Unverified'}
        </span>
      </header>

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
