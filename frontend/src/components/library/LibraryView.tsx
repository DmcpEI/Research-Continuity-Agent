import {
  type ChangeEvent,
  type FormEvent,
  type MouseEvent as ReactMouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ApiError, type ApiFetchResult, deleteSource, fetchPaper, ingestFile } from '../../api/client';
import { useSources } from '../../hooks/useSources';
import { SourceRow } from './SourceRow';

type LibraryViewProps = {
  onSourceOpen?: (sourceId: string) => void;
  selectedSourceId?: string | null;
};

export function LibraryView({ onSourceOpen, selectedSourceId = null }: LibraryViewProps) {
  const [filter, setFilter] = useState('');
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [identifier, setIdentifier] = useState('');
  const [fetchMessage, setFetchMessage] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<
    { sourceId: string; x: number; y: number } | null
  >(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const queryClient = useQueryClient();
  const { data: sources = [], isLoading, isError } = useSources();

  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => Promise.all(files.map((file) => ingestFile(file))),
    onSuccess: () => {
      setUploadMessage('Upload complete.');
      setUploadError(null);
      void queryClient.invalidateQueries({ queryKey: ['sources'] });
      void queryClient.invalidateQueries({ queryKey: ['status'] });
    },
    onError: (error) => {
      const message = errorText(error, 'Upload failed.');
      setUploadError(message);
      setUploadMessage(null);
    },
    onSettled: () => {
      setTimeout(() => setUploadMessage(null), 2500);
    },
  });

  const fetchMutation = useMutation({
    mutationFn: async (value: string) => fetchPaper(value),
    onSuccess: (result) => {
      setFetchMessage(fetchedText(result));
      setFetchError(null);
      setIdentifier('');
      void queryClient.invalidateQueries({ queryKey: ['sources'] });
      void queryClient.invalidateQueries({ queryKey: ['status'] });
    },
    onError: (error) => {
      setFetchError(errorText(error, 'Could not fetch the paper.'));
      setFetchMessage(null);
    },
  });

  const handleFetch = (event: FormEvent) => {
    event.preventDefault();
    const value = identifier.trim();
    if (!value || fetchMutation.isPending) {
      return;
    }
    setFetchError(null);
    setFetchMessage('Fetching…');
    fetchMutation.mutate(value);
  };

  const deleteMutation = useMutation({
    mutationFn: async (sourceId: string) => deleteSource(sourceId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['sources'] });
      void queryClient.invalidateQueries({ queryKey: ['status'] });
    },
  });

  const normalizedFilter = filter.trim().toLowerCase();
  const filteredSources = useMemo(() => {
    if (!normalizedFilter) {
      return sources;
    }

    return sources.filter((source) => {
      const title = source.title.toLowerCase();
      const id = source.id.toLowerCase();
      return title.includes(normalizedFilter) || id.includes(normalizedFilter);
    });
  }, [normalizedFilter, sources]);

  const totalChunks = useMemo(
    () => sources.reduce((sum, source) => sum + source.chunk_count, 0),
    [sources],
  );

  const handleAddSourcesClick = () => {
    setUploadError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';

    if (files.length === 0) {
      return;
    }

    const pdfs = files.filter(
      (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'),
    );

    if (pdfs.length === 0) {
      setUploadError('Please select one or more PDF files.');
      return;
    }

    setUploadError(null);
    setUploadMessage(`Uploading ${pdfs.length} PDF${pdfs.length === 1 ? '' : 's'}...`);
    uploadMutation.mutate(pdfs);
  };

  const handleContextMenu = (sourceId: string, event: ReactMouseEvent<HTMLTableRowElement>) => {
    setContextMenu({ sourceId, x: event.clientX, y: event.clientY });
  };

  const closeContextMenu = () => setContextMenu(null);

  const handleDeleteSource = () => {
    if (!contextMenu) {
      return;
    }
    const source = sources.find((item) => item.id === contextMenu.sourceId);
    const label = source?.title || source?.id || 'this source';
    const confirmed = window.confirm(`Delete ${label}? This cannot be undone.`);
    if (!confirmed) {
      closeContextMenu();
      return;
    }
    deleteMutation.mutate(contextMenu.sourceId, {
      onSettled: () => closeContextMenu(),
    });
  };

  useEffect(() => {
    if (!contextMenu) {
      return;
    }

    const handleClick = (event: globalThis.MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        closeContextMenu();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeContextMenu();
      }
    };

    window.addEventListener('mousedown', handleClick);
    window.addEventListener('keydown', handleEscape);
    return () => {
      window.removeEventListener('mousedown', handleClick);
      window.removeEventListener('keydown', handleEscape);
    };
  }, [contextMenu]);

  return (
    <section className="library-view">
      <header className="library-header">
        <div>
          <h1 className="library-title">Library</h1>
          <p className="library-subtitle">
            {sources.length} sources · {totalChunks} chunks
          </p>
        </div>
        <label className="library-filter-label" htmlFor="library-filter-input">
          <span>Filter</span>
          <input
            id="library-filter-input"
            className="library-filter-input"
            type="search"
            placeholder="Filter sources"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </label>
      </header>

      <div className="library-table-wrap">
        <table className="library-table">
          <thead>
            <tr>
              <th scope="col">SOURCE</th>
              <th scope="col">CHUNKS</th>
              <th scope="col">UPDATED</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr className="library-empty-row">
                <td colSpan={3}>Loading sources...</td>
              </tr>
            ) : null}

            {isError ? (
              <tr className="library-empty-row">
                <td colSpan={3}>Unable to load sources right now.</td>
              </tr>
            ) : null}

            {!isLoading && !isError && filteredSources.length === 0 ? (
              <tr className="library-empty-row">
                <td colSpan={3}>No sources match the current filter.</td>
              </tr>
            ) : null}

            {!isLoading && !isError
              ? filteredSources.map((source) => (
                  <SourceRow
                    key={source.id}
                    source={source}
                    onOpen={onSourceOpen}
                    selected={source.id === selectedSourceId}
                    onContextMenu={handleContextMenu}
                  />
                ))
              : null}

            <tr className="library-add-row">
              <td colSpan={3}>
                <button
                  className="library-add-button"
                  type="button"
                  onClick={handleAddSourcesClick}
                  disabled={uploadMutation.isPending}
                >
                  + Add sources
                </button>
                {uploadMessage ? (
                  <div className="library-upload-status">{uploadMessage}</div>
                ) : null}
                {uploadError ? (
                  <div className="library-upload-status is-error">{uploadError}</div>
                ) : null}
                <form className="library-fetch-form" onSubmit={handleFetch}>
                  <input
                    aria-label="arXiv ID or DOI"
                    className="library-filter-input library-fetch-input"
                    disabled={fetchMutation.isPending}
                    onChange={(event) => setIdentifier(event.target.value)}
                    placeholder="arXiv ID or DOI"
                    type="text"
                    value={identifier}
                  />
                  <button className="library-fetch-button" disabled={fetchMutation.isPending} type="submit">
                    Fetch
                  </button>
                </form>
                {fetchMessage ? <div className="library-upload-status">{fetchMessage}</div> : null}
                {fetchError ? <div className="library-upload-status is-error">{fetchError}</div> : null}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  multiple
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      {contextMenu ? (
        <div
          ref={menuRef}
          className="library-context-menu"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          role="menu"
        >
          <button
            type="button"
            className="library-context-item is-danger"
            role="menuitem"
            onClick={handleDeleteSource}
            disabled={deleteMutation.isPending}
          >
            Delete
          </button>
        </div>
      ) : null}
    </section>
  );
}

// The server's own message when there is one, not "API 404: {...}".
function errorText(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    return error.detail || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

function fetchedText(result: ApiFetchResult): string {
  const title = result.title || result.source_id;
  if (result.ingest_status === 'unchanged') {
    return `“${title}” is already in your library.`;
  }
  const where = result.origin.kind === 'arxiv' ? 'arXiv' : 'DOI';
  return `Added “${title}” (${where} ${result.origin.identifier}).`;
}
