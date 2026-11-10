/// <reference types="vite/client" />
import globalsCss from './globals.css?raw';

const sources = import.meta.glob(['../**/*.ts', '../**/*.tsx', '!../**/*.test.*'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const RAW_COLOUR =
  /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(|\b(?:white|black|red|green|blue|orange|yellow|purple|gr[ae]y)\b/i;
const RAW_LENGTH = /(?<![\w-])-?\d*\.?\d+(?:px|rem|em)\b/;
// Zero durations (reduced motion) are allowed; any other literal duration is raw.
const RAW_DURATION = /(?<![\w.-])(?!0*\.?0+m?s\b)\d*\.?\d+m?s\b/;
const SCALED_PROPERTIES =
  /^(?:font|font-size|font-family|letter-spacing|padding(?:-\w+)?|margin(?:-\w+)?|gap|row-gap|column-gap|border-radius|transition(?:-\w+)?|animation(?:-\w+)?)$/;

type Declaration = { property: string; value: string; line: number };

// A declaration starts after `{` or `;` and ends at `;` or `}`, so one-line rules and
// values that span lines are both seen. @import is dropped first: its URL holds a `;`.
function declarations(css: string): Declaration[] {
  const blank = (text: string) => text.replace(/[^\n]/g, ' ');
  const cleaned = css.replace(/\/\*[\s\S]*?\*\//g, blank).replace(/@import[^\n]*/g, blank);
  const found: Declaration[] = [];
  for (const match of cleaned.matchAll(/[{;]\s*([a-zA-Z-]+)\s*:\s*([^;{}]+?)\s*(?=[;}])/g)) {
    const start = (match.index ?? 0) + match[0].indexOf(match[1]);
    found.push({
      property: match[1].toLowerCase(),
      value: match[2].replace(/\s+/g, ' '),
      line: cleaned.slice(0, start).split('\n').length,
    });
  }
  return found;
}

const describeAll = (found: Declaration[]) =>
  found.map(({ property, value, line }) => `globals.css:${line} ${property}: ${value}`);

function rawColours(css: string): string[] {
  return describeAll(declarations(css).filter(({ value }) => RAW_COLOUR.test(value)));
}

function offScale(css: string): string[] {
  return describeAll(
    declarations(css)
      .filter(({ property }) => SCALED_PROPERTIES.test(property))
      .filter(({ value }) => RAW_LENGTH.test(value) || RAW_DURATION.test(value) || /'IBM Plex/.test(value)),
  );
}

describe('design tokens', () => {
  it('keeps every colour in globals.css behind a token', () => {
    expect(rawColours(globalsCss)).toEqual([]);
  });

  it('keeps type, spacing, radius and motion in globals.css on the scale', () => {
    expect(offScale(globalsCss)).toEqual([]);
  });

  it('keeps colour literals out of components', () => {
    const offenders = Object.entries(sources).flatMap(([path, text]) =>
      text
        .split('\n')
        .map((line, index) => ({ line, index }))
        .filter(({ line }) => /['"`]#[0-9a-f]{3,8}['"`]/i.test(line))
        .map(({ index }) => `${path}:${index + 1}`),
    );
    expect(offenders).toEqual([]);
  });

  it('sees raw values however a declaration is written', () => {
    const css = [
      '.a { color: #ff0000; }',
      '.b {\n  box-shadow: 0 0 0 1px var(--border),\n    0 2px 4px rgba(0, 0, 0, 0.5);\n}',
      '.c { color: red; background: oklch(70% 0.1 200); }',
      '.d { FONT: 600 10px/1.2 monospace; }',
      '.e { animation: spin 900ms linear; animation-delay: 0.2s; transition-duration: 300ms; }',
    ].join('\n');
    expect(rawColours(css)).toHaveLength(4);
    expect(offScale(css)).toHaveLength(4);
  });

  it('accepts tokens, zero durations and the font import', () => {
    const css = [
      "@import url('https://example.test/css?family=A:wght@400;600&display=swap');",
      '.a:hover { color: var(--text); transition: color var(--t-fast); }',
      '.b { animation-duration: 0ms; transition-duration: 0s; animation-delay: calc(var(--t-loop) / 5); }',
    ].join('\n');
    expect(rawColours(css)).toEqual([]);
    expect(offScale(css)).toEqual([]);
  });
});
