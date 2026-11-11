import type { ApiSourceSummary } from '../../api/client';

// Source titles are often filenames ("AutoBag_Learning_to_Open_..."); show them as words.
export function displayTitle(title: string): string {
  return title.replace(/_+/g, ' ').replace(/\s+/g, ' ').trim();
}

// ponytail: filenames are often cut at 50 or 90 characters and some have no spaces at all, so
// prefer short titles made of real words; better titles at ingest (DESIGN F7) make this moot.
function isReadable(title: string): boolean {
  return title.length <= 80 && title.split(' ').every((word) => word.length <= 20);
}

function pickTitles(sources: ApiSourceSummary[], count: number): string[] {
  const titles = [...new Set(sources.map((source) => displayTitle(source.title)).filter(Boolean))];
  const readable = titles.filter(isReadable);
  return (readable.length > 0 ? readable : titles).slice(0, count);
}

// Questions RCA can answer from this library (checked live: answered with citations).
export function chatSuggestions(sources: ApiSourceSummary[]): string[] {
  return pickTitles(sources, 3).map((title) => `What problem does “${title}” address, and how?`);
}

// One knowledge-base task from the library plus two that work on any setup.
export function agentExamples(sources: ApiSourceSummary[]): string[] {
  return [
    ...pickTitles(sources, 1).map(
      (title) => `Search the knowledge base for “${title}” and say in one sentence what it is about.`,
    ),
    'List the files in the workspace folder.',
    'List the recorded experiment runs and their statuses.',
  ];
}
