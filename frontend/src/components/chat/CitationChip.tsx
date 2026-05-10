type CitationChipProps = {
  sourceId: string;
  onClick?: (sourceId: string) => void;
};

export function CitationChip({ sourceId, onClick }: CitationChipProps) {
  return (
    <button type="button" className="citation-chip" onClick={() => onClick?.(sourceId)}>
      {sourceId}
    </button>
  );
}
