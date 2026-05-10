import { Link, useLocation } from 'react-router-dom';
import { useStatus } from '../../hooks/useStatus';

export function LeftRail() {
  const location = useLocation();
  const { data, isLoading } = useStatus();

  const papers = isLoading ? '--' : String(data?.papers ?? '--');
  const chunks = isLoading ? '--' : String(data?.chunks ?? '--');

  return (
    <aside className="left-rail">
      <div className="left-rail-brand">RCA</div>
      <nav className="left-rail-nav">
        <Link aria-label="Chat" className={location.pathname === '/' ? 'active' : ''} to="/">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M6 6h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-6l-4 3v-3H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinejoin="round"
            />
          </svg>
        </Link>
        <Link
          aria-label="Library"
          className={location.pathname === '/library' ? 'active' : ''}
          to="/library"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="4" y="4" width="6" height="6" rx="1" fill="none" stroke="currentColor" strokeWidth="1.7" />
            <rect x="14" y="4" width="6" height="6" rx="1" fill="none" stroke="currentColor" strokeWidth="1.7" />
            <rect x="4" y="14" width="6" height="6" rx="1" fill="none" stroke="currentColor" strokeWidth="1.7" />
            <rect x="14" y="14" width="6" height="6" rx="1" fill="none" stroke="currentColor" strokeWidth="1.7" />
          </svg>
        </Link>
        <Link
          aria-label="Agent"
          className={location.pathname === '/agent' ? 'active' : ''}
          to="/agent"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="5" y="7" width="14" height="10" rx="3" fill="none" stroke="currentColor" strokeWidth="1.7" />
            <circle cx="9.5" cy="12" r="1" fill="currentColor" />
            <circle cx="14.5" cy="12" r="1" fill="currentColor" />
            <path d="M12 4v2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
        </Link>
        <Link
          aria-label="Settings"
          className={location.pathname === '/settings' ? 'active' : ''}
          to="/settings"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M9.2 4.2h5.6l.5 2a6.8 6.8 0 0 1 1.6.9l1.9-.8 2.8 2.8-.8 1.9c.4.5.7 1 .9 1.6l2 .5v5.6l-2 .5a6.8 6.8 0 0 1-.9 1.6l.8 1.9-2.8 2.8-1.9-.8c-.5.4-1 .7-1.6.9l-.5 2H9.2l-.5-2a6.8 6.8 0 0 1-1.6-.9l-1.9.8-2.8-2.8.8-1.9a6.8 6.8 0 0 1-.9-1.6l-2-.5v-5.6l2-.5c.2-.6.5-1.1.9-1.6l-.8-1.9 2.8-2.8 1.9.8c.5-.4 1-.7 1.6-.9l.5-2Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
            <circle cx="12" cy="12" r="2.3" fill="none" stroke="currentColor" strokeWidth="1.4" />
          </svg>
        </Link>
      </nav>
      <div className="status-pill">{papers} papers · {chunks} chunks</div>
    </aside>
  );
}
