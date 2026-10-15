/**
 * Per-tag accent, derived from the tag name.
 *
 * The reference site lets tags carry colour, and it stays tasteful because
 * only the hue moves: saturation and lightness are pinned, so every chip sits
 * at the same chroma and the set reads as one family. The failure mode to avoid
 * is rotating a palette by index -- that is what makes a tag cloud look cheap,
 * because colour then encodes nothing and two unrelated tags collide.
 *
 * Hue is spread across the full wheel on purpose. Confining it to the brand's
 * teal, as the series bands do, would turn every chip into the same green and
 * cost the reader the ability to tell tags apart at a glance. Chroma is what
 * has to be controlled, not hue.
 */

/** Stable hue in [0, 360) for any string. */
export function tagHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) >>> 0;
  }
  return h % 360;
}

/** Inline style carrying the two pinned tones for a tag chip. */
export function tagChipStyle(name: string): string {
  const h = tagHue(name);
  return `--tag-h: ${h};`;
}