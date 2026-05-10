import { useState, type CSSProperties, type PropsWithChildren } from 'react';
import { LeftRail } from './LeftRail';
import { RightPanel } from './RightPanel';
import { StatusBar } from './StatusBar';

type AppShellProps = PropsWithChildren<{
  graphFocusSourceIds: string[];
  selectedSourceId: string | null;
  showRightPanel: boolean;
  sourceOnlyRightPanel: boolean;
}>;

export function AppShell({
  children,
  graphFocusSourceIds,
  selectedSourceId,
  showRightPanel,
  sourceOnlyRightPanel,
}: AppShellProps) {
  const [rightPanelWidth, setRightPanelWidth] = useState(284);
  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState(false);

  return (
    <div
      className={`app-shell ${showRightPanel ? '' : 'no-right-panel'}`.trim()}
      style={
        {
          '--right-panel-width': isRightPanelCollapsed ? '22px' : `${rightPanelWidth}px`,
        } as CSSProperties
      }
    >
      <LeftRail />
      <main className="app-main">{children}</main>
      {showRightPanel ? (
        <RightPanel
          graphFocusSourceIds={graphFocusSourceIds}
          isCollapsed={isRightPanelCollapsed}
          selectedSourceId={selectedSourceId}
          setIsCollapsed={setIsRightPanelCollapsed}
          rightPanelWidth={rightPanelWidth}
          setRightPanelWidth={setRightPanelWidth}
          sourceOnly={sourceOnlyRightPanel}
        />
      ) : null}
      <StatusBar />
    </div>
  );
}
