import { useEffect, useMemo, useState } from 'react';

type QueryTraceProps = {
  trace: Record<string, unknown> | null;
};

type TraceStageRow = {
  label: string;
  detail: string;
  durationMs: number | null;
};

type StageEntry = {
  name: string;
  duration_ms: number;
  hit_count?: number;
  notes?: string;
};

function toStageEntries(trace: Record<string, unknown>): StageEntry[] {
  const rawStages = trace.stages;
  if (!Array.isArray(rawStages)) {
    return [];
  }

  return rawStages
    .map((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        return null;
      }

      const name = (item as Record<string, unknown>).name;
      const durationMs = (item as Record<string, unknown>).duration_ms;
      const hitCount = (item as Record<string, unknown>).hit_count;
      const notes = (item as Record<string, unknown>).notes;

      if (typeof name !== 'string' || typeof durationMs !== 'number') {
        return null;
      }

      return {
        name,
        duration_ms: durationMs,
        hit_count: typeof hitCount === 'number' ? hitCount : undefined,
        notes: typeof notes === 'string' ? notes : undefined,
      } as StageEntry;
    })
    .filter((stage): stage is StageEntry => stage !== null);
}

function inferHitCount(trace: Record<string, unknown>): number | null {
  const stages = toStageEntries(trace);
  for (let index = stages.length - 1; index >= 0; index -= 1) {
    const count = stages[index]?.hit_count;
    if (typeof count === 'number') {
      return count;
    }
  }

  const directHits = trace.hits;
  if (typeof directHits === 'number') {
    return directHits;
  }

  const contextNodes = trace.context_node_ids;
  if (Array.isArray(contextNodes)) {
    return contextNodes.length;
  }

  const provenance = trace.provenance;
  if (Array.isArray(provenance)) {
    return provenance.length;
  }

  return null;
}

function inferLatencySeconds(trace: Record<string, unknown>): string {
  const raw = trace.total_latency_ms;
  if (typeof raw !== 'number') {
    return '--';
  }
  return (raw / 1000).toFixed(2);
}

function toStageName(key: string): string {
  return key
    .replace(/_latency_ms$/, '')
    .replace(/_ms$/, '')
    .replace(/_/g, ' ');
}

function titleCase(value: string): string {
  return value
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function buildTraceRows(trace: Record<string, unknown>): TraceStageRow[] {
  const stages = toStageEntries(trace);
  if (stages.length > 0) {
    return stages.map((stage) => {
      let detail = '--';
      if (typeof stage.hit_count === 'number') {
        detail = `${stage.hit_count} hits`;
      }
      if (stage.notes && stage.notes.trim()) {
        detail = detail === '--' ? stage.notes : `${detail} · ${stage.notes}`;
      }

      return {
        label: titleCase(stage.name),
        detail,
        durationMs: stage.duration_ms,
      };
    });
  }

  return Object.entries(trace)
    .filter(([key, value]) => key !== 'total_latency_ms' && (typeof value === 'string' || typeof value === 'number'))
    .slice(0, 6)
    .map(([key, value]) => ({
      label: toStageName(key),
      detail: String(value),
      durationMs: null,
    }));
}

function extractScores(trace: Record<string, unknown>): string {
  const scores = trace.scores;
  if (!scores || typeof scores !== 'object' || Array.isArray(scores)) {
    return '';
  }

  return Object.entries(scores)
    .slice(0, 5)
    .map(([key, value]) => `${key} ${typeof value === 'number' ? value.toFixed(2) : String(value)}`)
    .join('   ');
}

export function QueryTrace({ trace }: QueryTraceProps) {
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setExpanded(false);
  }, [trace]);

  const summary = useMemo(() => {
    if (!trace || Object.keys(trace).length === 0) {
      return null;
    }

    const hits = inferHitCount(trace);
    const latency = inferLatencySeconds(trace);
    const stageCount = toStageEntries(trace).length;
    return `${hits ?? '--'} hits · ${stageCount || '--'} stages · ${latency}s`;
  }, [trace]);

  const rows = useMemo(() => {
    if (!trace) {
      return [];
    }
    return buildTraceRows(trace);
  }, [trace]);

  const scoreLine = useMemo(() => {
    if (!trace) {
      return '';
    }
    return extractScores(trace);
  }, [trace]);

  if (!trace || !summary) {
    return null;
  }

  return (
    <div className="query-trace">
      <button className="query-trace-toggle" onClick={() => setExpanded((value) => !value)} type="button">
        <span className="query-trace-chevron">{expanded ? '▾' : '›'}</span>
        <span>{summary}</span>
      </button>
      {expanded ? (
        <div className="query-trace-panel">
          <div className="query-trace-rows">
            {rows.map((row) => (
              <div className="query-trace-row" key={row.label}>
                <span className="query-trace-row-left">{row.label}</span>
                <span className="query-trace-row-mid">{row.detail}</span>
                <span className="query-trace-row-right">
                  {row.durationMs !== null ? `${row.durationMs.toFixed(2)}ms` : '--'}
                </span>
              </div>
            ))}
          </div>
          {scoreLine ? <div className="query-trace-score-line">scores: {scoreLine}</div> : null}
        </div>
      ) : null}
    </div>
  );
}
