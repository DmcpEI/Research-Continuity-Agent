import { getPdfUrl } from '../../api/client';

type PdfViewerProps = {
  sourceId: string;
};

type OpenPdfInNewTabButtonProps = {
  sourceId: string;
};

export function OpenPdfInNewTabButton({ sourceId }: OpenPdfInNewTabButtonProps) {
  if (!sourceId) {
    return null;
  }

  const url = getPdfUrl(sourceId);

  return (
    <button
      type="button"
      className="right-panel-open-tab-btn"
      onClick={() => window.open(url, '_blank')}
      title="Open PDF in new tab"
      aria-label="Open PDF in new tab"
    >
      ↗
    </button>
  );
}

export function PdfViewer({ sourceId }: PdfViewerProps) {
  if (!sourceId) {
    return <div className="pdf-viewer-empty">Select a source to preview PDF</div>;
  }

  return (
    <div className="pdf-viewer">
      <iframe className="pdf-viewer-frame" src={getPdfUrl(sourceId)} title="PDF viewer" />
    </div>
  );
}
