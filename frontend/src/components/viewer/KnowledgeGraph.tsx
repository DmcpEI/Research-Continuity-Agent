import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { ApiSourceSummary } from '../../api/client';
import { useSources } from '../../hooks/useSources';

type GraphNode = d3.SimulationNodeDatum & {
  id: string;
  title: string;
  kind: string;
};

type GraphLink = d3.SimulationLinkDatum<GraphNode> & {
  source: string | GraphNode;
  target: string | GraphNode;
};

function kindColor(kind: string): string {
  if (kind === 'note') {
    return '#22c55e';
  }
  if (kind === 'experiment') {
    return '#f59e0b';
  }
  if (kind === 'paper' || kind === 'pdf') {
    return '#7c6af7';
  }
  return '#a0a0b8';
}

function nodeLabel(sourceId: string): string {
  const parts = sourceId.split('/');
  const shortId = parts[parts.length - 1] || sourceId;
  return shortId.length <= 12 ? shortId : `${shortId.slice(0, 12)}…`;
}

function shortSourceId(sourceId: string): string {
  const parts = sourceId.split('/');
  return parts[parts.length - 1] || sourceId;
}

function extractConceptsFromTitles(titles: string[]): string[] {
  const stopWords = new Set([
    'the',
    'and',
    'for',
    'with',
    'from',
    'into',
    'using',
    'based',
    'study',
    'paper',
    'towards',
    'approach',
    'methods',
  ]);

  const tokenCounts = new Map<string, number>();
  for (const title of titles) {
    const tokens = title
      .toLowerCase()
      .split(/[^a-z0-9-]+/)
      .filter((token) => token.length > 3 && !stopWords.has(token));
    for (const token of tokens) {
      tokenCounts.set(token, (tokenCounts.get(token) ?? 0) + 1);
    }
  }

  return Array.from(tokenCounts.entries())
    .sort((left, right) => right[1] - left[1])
    .slice(0, 6)
    .map(([token]) => token);
}

function buildLinks(sources: ApiSourceSummary[]): GraphLink[] {
  const byKind = new Map<string, ApiSourceSummary[]>();
  for (const source of sources) {
    const bucket = byKind.get(source.kind) ?? [];
    bucket.push(source);
    byKind.set(source.kind, bucket);
  }

  const links: GraphLink[] = [];
  for (const sameKindSources of byKind.values()) {
    for (let i = 0; i < sameKindSources.length; i += 1) {
      for (let j = i + 1; j < sameKindSources.length; j += 1) {
        links.push({ source: sameKindSources[i].id, target: sameKindSources[j].id });
      }
    }
  }
  return links;
}

type KnowledgeGraphProps = {
  focusSourceIds?: string[];
  selectedSourceId?: string | null;
};

export function KnowledgeGraph({ focusSourceIds = [], selectedSourceId = null }: KnowledgeGraphProps) {
  const { data: sources = [] } = useSources();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const focusSet = new Set(focusSourceIds);
  const focusedSources = sources.filter((source) => focusSet.has(source.id));
  const maxChunkCount = Math.max(1, ...focusedSources.map((source) => source.chunk_count));
  const activeSourceRows = focusedSources
    .map((source) => ({
      id: source.id,
      shortId: shortSourceId(source.id),
      score: Math.max(0.35, source.chunk_count / maxChunkCount),
      chunkCount: source.chunk_count,
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 3);

  const conceptSummary =
    focusedSources.length > 0
      ? extractConceptsFromTitles(focusedSources.map((source) => source.title)).length > 0
        ? extractConceptsFromTitles(focusedSources.map((source) => source.title))
        : Array.from(new Set(focusedSources.map((source) => source.kind))).slice(0, 6)
      : [];

  useEffect(() => {
    if (!svgRef.current || sources.length === 0) {
      return;
    }

    const svgEl = svgRef.current;
    const width = svgEl.clientWidth || 280;
    const height = svgEl.clientHeight || 260;

    const nodes: GraphNode[] = sources.map((source) => ({
      id: source.id,
      title: source.title,
      kind: source.kind,
    }));
    const links = buildLinks(sources);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const linkSelection = svg
      .append('g')
      .attr('stroke', '#2c2c3a')
      .attr('stroke-opacity', 0.9)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', 1);

    const nodeSelection = svg
      .append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (node: GraphNode) => {
        if (selectedSourceId && node.id === selectedSourceId) {
          return 10;
        }
        if (focusSet.has(node.id)) {
          return 7;
        }
        return 5;
      })
      .attr('fill', (node: GraphNode) => kindColor(node.kind))
      .attr('opacity', (node: GraphNode) => {
        if (focusSet.size === 0) {
          return 0.9;
        }
        if (selectedSourceId && node.id === selectedSourceId) {
          return 1;
        }
        return focusSet.has(node.id) ? 0.95 : 0.32;
      });

    nodeSelection.append('title').text((node: GraphNode) => node.title);

    const labels = svg
      .append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text((node: GraphNode) => nodeLabel(node.id))
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', 9)
      .attr('fill', (node: GraphNode) =>
        selectedSourceId && node.id === selectedSourceId ? '#e2e2e8' : '#a0a0b8',
      )
      .attr('text-anchor', 'middle')
      .attr('opacity', (node: GraphNode) => (focusSet.size === 0 || focusSet.has(node.id) ? 0.95 : 0.28))
      .attr('pointer-events', 'none');

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((node: GraphNode) => node.id)
          .distance(60),
      )
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2));

    simulation.on('tick', () => {
      linkSelection
        .attr('x1', (link: GraphLink) => (link.source as GraphNode).x ?? 0)
        .attr('y1', (link: GraphLink) => (link.source as GraphNode).y ?? 0)
        .attr('x2', (link: GraphLink) => (link.target as GraphNode).x ?? 0)
        .attr('y2', (link: GraphLink) => (link.target as GraphNode).y ?? 0);

      nodeSelection
        .attr('cx', (node: GraphNode) => node.x ?? 0)
        .attr('cy', (node: GraphNode) => node.y ?? 0);

      labels
        .attr('x', (node: GraphNode) => node.x ?? 0)
        .attr('y', (node: GraphNode) => (node.y ?? 0) + 12);
    });

    return () => {
      simulation.stop();
    };
  }, [focusSet, selectedSourceId, sources]);

  return (
    <div className="knowledge-graph">
      <div className="knowledge-graph-canvas-wrap">
        <svg className="knowledge-graph-canvas" ref={svgRef} role="img" aria-label="Knowledge graph" />
      </div>

      <section className="knowledge-graph-section">
        <h3>Active Sources</h3>
        {activeSourceRows.length > 0 ? (
          <div className="knowledge-graph-source-list">
            {activeSourceRows.map((source) => (
              <div className="knowledge-graph-source-item" key={source.id}>
                <span className="knowledge-graph-source-title">
                  <span className="knowledge-graph-source-dot" aria-hidden="true" />
                  {source.shortId}
                </span>
                <span className="knowledge-graph-source-kind">
                  {source.score.toFixed(2)} <span>x{source.chunkCount}</span>
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p>No sources active for this conversation yet.</p>
        )}
      </section>

      <section className="knowledge-graph-section">
        <h3>Concepts</h3>
        {conceptSummary.length > 0 ? (
          <div className="knowledge-graph-concept-list">
            {conceptSummary.map((concept) => (
              <span className="knowledge-graph-concept-chip" key={concept}>
                {concept}
              </span>
            ))}
          </div>
        ) : (
          <p>No concepts extracted yet.</p>
        )}
      </section>
    </div>
  );
}
