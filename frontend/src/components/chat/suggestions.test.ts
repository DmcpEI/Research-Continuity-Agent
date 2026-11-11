import type { ApiSourceSummary } from '../../api/client';
import { agentExamples, chatSuggestions, displayTitle } from './suggestions';

function source(id: string, title: string): ApiSourceSummary {
  return {
    id,
    title,
    kind: 'paper',
    chunk_count: 3,
    latest_path: null,
    content_sha256: null,
    latest_revision_created_at: null,
  };
}

describe('displayTitle', () => {
  it('turns a filename-style title into words', () => {
    expect(displayTitle('Robotic_Grocery_Bagging__Structured_Perception ')).toBe(
      'Robotic Grocery Bagging Structured Perception',
    );
    expect(displayTitle('AI-Powered_Robotic_Systems_for_Intelligent_Retail_')).toBe(
      'AI-Powered Robotic Systems for Intelligent Retail',
    );
  });
});

describe('chatSuggestions', () => {
  it('asks about up to three library titles, preferring readable ones', () => {
    const sources = [
      source('src:pdf/a', 'LLM-Pack_Intuitive_Grocery_Handling_for_Logistics_Applications'),
      source('src:pdf/b', `Language-Grounded_${'Very_Long_Title_'.repeat(6)}Manipulat`),
      source('src:pdf/c', 'Restockandstraighteningsystemforretailautomationusingcompliant'),
      source('src:pdf/d', 'AutoBag_Learning_to_Open_Plastic_Bags_and_Insert_Objects'),
      source('src:pdf/e', 'Grammar-Aligned_Decoding'),
      source('src:pdf/f', 'JamPacker'),
    ];
    expect(chatSuggestions(sources)).toEqual([
      'What problem does “LLM-Pack Intuitive Grocery Handling for Logistics Applications” address, and how?',
      'What problem does “AutoBag Learning to Open Plastic Bags and Insert Objects” address, and how?',
      'What problem does “Grammar-Aligned Decoding” address, and how?',
    ]);
  });

  it('falls back to any titles when none is readable, and skips blank or repeated ones', () => {
    const long = `Language-Grounded_${'Very_Long_Title_'.repeat(6)}`;
    const sources = [source('src:pdf/a', long), source('src:pdf/b', '  '), source('src:pdf/c', long)];
    expect(chatSuggestions(sources)).toEqual([`What problem does “${displayTitle(long)}” address, and how?`]);
  });

  it('suggests nothing for an empty library', () => {
    expect(chatSuggestions([])).toEqual([]);
  });
});

describe('agentExamples', () => {
  it('pairs one knowledge-base task from the library with tasks that work on any setup', () => {
    expect(agentExamples([source('src:pdf/a', 'Grammar-Aligned_Decoding')])).toEqual([
      'Search the knowledge base for “Grammar-Aligned Decoding” and say in one sentence what it is about.',
      'List the files in the workspace folder.',
      'List the recorded experiment runs and their statuses.',
    ]);
    expect(agentExamples([])).toEqual([
      'List the files in the workspace folder.',
      'List the recorded experiment runs and their statuses.',
    ]);
  });
});
