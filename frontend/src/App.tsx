import { useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { AppShell } from './components/shell/AppShell';
import { ChatView } from './components/chat/ChatView';
import { LibraryView } from './components/library/LibraryView';
import { AgentView } from './components/agent/AgentView';

export default function App() {
  const location = useLocation();
  const [chatSelectedSourceId, setChatSelectedSourceId] = useState<string | null>(null);
  const [librarySelectedSourceId, setLibrarySelectedSourceId] = useState<string | null>(null);
  const [chatGraphFocusSourceIds, setChatGraphFocusSourceIds] = useState<string[]>([]);

  const pathname = location.pathname;
  const isKnownRoute = pathname === '/' || pathname === '/library' || pathname === '/agent' || pathname === '/settings';
  if (!isKnownRoute) {
    return <Navigate to="/" replace />;
  }

  const selectedSourceId =
    pathname === '/'
      ? chatSelectedSourceId
      : pathname === '/library'
        ? librarySelectedSourceId
          : null;

  const graphFocusSourceIds = pathname === '/' ? chatGraphFocusSourceIds : [];

  const showRightPanel =
    pathname === '/' ||
    (pathname === '/library' && Boolean(librarySelectedSourceId));
  const sourceOnlyRightPanel = pathname === '/library' && Boolean(librarySelectedSourceId);

  return (
    <AppShell
      graphFocusSourceIds={graphFocusSourceIds}
      selectedSourceId={selectedSourceId}
      showRightPanel={showRightPanel}
      sourceOnlyRightPanel={sourceOnlyRightPanel}
    >
      <div className={`app-route-pane ${pathname === '/' ? 'is-visible' : 'is-hidden'}`}>
        <ChatView
          onCitationClick={setChatSelectedSourceId}
          onGraphFocusChange={setChatGraphFocusSourceIds}
        />
      </div>
      <div className={`app-route-pane ${pathname === '/library' ? 'is-visible' : 'is-hidden'}`}>
        <LibraryView
          onSourceOpen={setLibrarySelectedSourceId}
          selectedSourceId={librarySelectedSourceId}
        />
      </div>
      <div className={`app-route-pane ${pathname === '/agent' ? 'is-visible' : 'is-hidden'}`}>
        <AgentView conversationId="agent-main" />
      </div>
      <div className={`app-route-pane ${pathname === '/settings' ? 'is-visible' : 'is-hidden'}`}>
        <section className="view-placeholder" aria-label="Settings">
          Settings panel is coming soon.
        </section>
      </div>
    </AppShell>
  );
}
