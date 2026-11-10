/// <reference types="vite/client" />
import globalsCss from './globals.css?raw';

const sources = import.meta.glob(['../**/*.ts', '../**/*.tsx', '!../**/*.test.*'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

// A tripwire for the usual forms, not a full CSS colour parser: hex (also `%23` in data URIs),
// colour functions and the common names. `url(#id)` is a reference, not a colour.
const RAW_COLOUR =
  /(?<!url\()(?:#|%23)[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(|(?<![\w-])(?:white|black|red|green|blue|orange|yellow|purple|gr[ae]y)(?![\w-])/i;
const RAW_LENGTH =
  /(?<![\w-])-?\d*\.?\d+(?:px|rem|em|pt|pc|cm|mm|in|q|vh|vw|vmin|vmax|ch|ex|%)(?![\w-])|(?<![\w-])(?:xx?-small|small|medium|large|xx?-large|smaller|larger)(?![\w-])/i;
// Zero durations (reduced motion) are allowed; any other literal duration is raw.
const RAW_DURATION = /(?<![\w.-])(?!0*\.?0+m?s\b)\d*\.?\d+(?:e[+-]?\d+)?m?s\b/i;
// Custom properties count too: a token defined outside tokens.css is a raw value with a name.
const SCALED_PROPERTIES =
  /^(?:--[\w-]+|font|font-size|font-family|letter-spacing|padding(?:-\w+)?|margin(?:-\w+)?|gap|row-gap|column-gap|border-radius|transition(?:-\w+)?|animation(?:-\w+)?)$/;

type Declaration = { property: string; value: string; line: number };

// A declaration starts after `{`, `;` or `}` and ends at `;` or `}`, so one-line rules, nested
// rules and values that span lines are all seen. `;{}` inside url() or strings are not
// separators, and @import is dropped up to its own `;`.
function declarations(css: string): Declaration[] {
  const blank = (text: string) => text.replace(/[^\n]/g, ' ');
  const cleaned = css
    .replace(/\/\*[\s\S]*?\*\//g, blank)
    .replace(/url\([^)]*\)|"[^"\n]*"|'[^'\n]*'/g, (text) => text.replace(/[;{}]/g, ' '))
    .replace(/@import[^;\n]*;/g, blank);
  const found: Declaration[] = [];
  for (const match of cleaned.matchAll(
    /[{};]\s*(-{0,2}[a-zA-Z][\w-]*)\s*:\s*([^;{}]+?)\s*(?=[;}])/g,
  )) {
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
  return describeAll(
    declarations(css).filter(({ property, value }) => property !== 'content' && RAW_COLOUR.test(value)),
  );
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

  it.each([
    ['one-line rule', '.a { color: #ff0000; }'],
    ['value over several lines', '.a {\n  box-shadow: 0 0 0 1px var(--border),\n    0 2px 4px rgba(0, 0, 0, 0.5);\n}'],
    ['named colour', '.a { color: red; }'],
    ['oklch()', '.a { background: oklch(70% 0.1 200); }'],
    ['custom property with a digit', '.a { --text-4: #ff0000; }'],
    ['after an @import on the same line', '@import url(x.css); .a { color: #f00; }'],
    ['after a nested rule', '.a { .b { gap: var(--s-1); } color: #f00; }'],
    ['inside a data URI', `.a { background: url("data:image/svg+xml;utf8,<svg fill='#ff0000'/>"); }`],
    ['percent-encoded in a data URI', `.a { background: url("data:image/svg+xml;utf8,<svg fill='%23ff0000'/>"); }`],
  ])('sees a raw colour: %s', (_, css) => {
    expect(rawColours(css)).toHaveLength(1);
  });

  it.each([
    ['font shorthand, any case', '.a { FONT: 600 10px/1.2 monospace; }'],
    ['animation duration', '.a { animation: spin 900ms linear; }'],
    ['animation delay', '.a { animation-delay: 0.2s; }'],
    ['exponent duration', '.a { animation: pulse 1e3ms; }'],
    ['transition-duration', '.a { transition-duration: 300ms; }'],
    ['pt', '.a { font-size: 9pt; }'],
    ['percent font size', '.a { font-size: 70%; }'],
    ['keyword font size', '.a { font-size: x-small; }'],
    ['viewport unit', '.a { padding: 2vw; }'],
    ['custom property with a digit', '.a { --s-5: 20px; padding: var(--s-5); }'],
  ])('sees an off-scale value: %s', (_, css) => {
    expect(offScale(css)).toHaveLength(1);
  });

  it('accepts tokens, zero durations, the font import and colour words in names', () => {
    const css = [
      "@import url('https://example.test/css?family=A:wght@400;600&display=swap');",
      '.a:hover { color: var(--text); transition: color var(--t-fast); }',
      '.b { animation-duration: 0ms; transition-duration: 0s; animation-delay: calc(var(--t-loop) / 5); }',
      '.c { color: var(--green); animation: red-pulse var(--t-loop); grid-area: blue-box; }',
      '.d { fill: url(#fade); content: "Black box"; border-radius: var(--r-full); }',
    ].join('\n');
    expect(rawColours(css)).toEqual([]);
    expect(offScale(css)).toEqual([]);
  });
});
