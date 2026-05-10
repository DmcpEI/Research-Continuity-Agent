import { useEffect, useState } from 'react';
import { KnowledgeGraph } from '../viewer/KnowledgeGraph';
import { OpenPdfInNewTabButton, PdfViewer } from '../viewer/PdfViewer';

type RightPanelProps = {
  graphFocusSourceIds: string[];
  isCollapsed: boolean;
  selectedSourceId: string | null;
  setIsCollapsed: (collapsed: boolean) => void;
  rightPanelWidth: number;
  setRightPanelWidth: (width: number) => void;
  sourceOnly?: boolean;
};

export function RightPanel({
  graphFocusSourceIds,
  isCollapsed,
  selectedSourceId,
  setIsCollapsed,
  rightPanelWidth,
  setRightPanelWidth,
  sourceOnly = false,
}: RightPanelProps) {
  const [tab, setTab] = useState<'graph' | 'source'>(sourceOnly ? 'source' : 'graph');

  const handleResizeStart = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();

    const startX = event.clientX;
    const startWidth = rightPanelWidth;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const nextWidth = startWidth + (startX - moveEvent.clientX);
      setRightPanelWidth(Math.min(600, Math.max(240, nextWidth)));
    };

    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  useEffect(() => {
    if (sourceOnly) {
      setTab('source');
      return;
    }

    if (selectedSourceId) {
      setTab('source');
      return;
    }
    setTab('graph');
  }, [selectedSourceId, sourceOnly]);

  if (isCollapsed) {
    return (
      <aside className="right-panel is-collapsed" aria-label="Collapsed right panel">
        <button
          aria-label="Show right panel"
          className="right-panel-collapse-btn"
          type="button"
          onClick={() => setIsCollapsed(false)}
        >
          ‹
        </button>
      </aside>
    );
  }

  return (
    <aside className="right-panel">
      <div className="right-panel-resize-handle" onMouseDown={handleResizeStart} />
      <div className="right-panel-tabs">
        {!sourceOnly ? (
          <button
            className={tab === 'graph' ? 'active' : ''}
            type="button"
            onClick={() => setTab('graph')}
          >
            Graph
          </button>
        ) : null}
        <button
          className={tab === 'source' ? 'active' : ''}
          type="button"
          onClick={() => setTab('source')}
        >
          Source
        </button>
        <button
          aria-label="Hide right panel"
          className="right-panel-collapse-btn"
          type="button"
          onClick={() => setIsCollapsed(true)}
        >
          ›
        </button>
        {tab === 'source' && selectedSourceId ? (
          <OpenPdfInNewTabButton sourceId={selectedSourceId} />
        ) : null}
      </div>
      <div className="right-panel-content">
        {sourceOnly ? (
          selectedSourceId ? (
            <PdfViewer sourceId={selectedSourceId} />
          ) : (
            <div className="pdf-viewer-empty">Select a citation to open a source PDF.</div>
          )
        ) : tab === 'graph' ? (
          <KnowledgeGraph
            focusSourceIds={graphFocusSourceIds}
            selectedSourceId={selectedSourceId}
          />
        ) : selectedSourceId ? (
          <PdfViewer sourceId={selectedSourceId} />
        ) : (
          <div className="pdf-viewer-empty">Select a citation to open a source PDF.</div>
        )}
      </div>
    </aside>
  );
}
