/**
 * Deterministic accent hue for a series.
 *
 * Every series owns one accent that shows up on its list rows and its own
 * index card. The hue is confined to a narrow window around the brand hue
 * (94–154) so six series read as one colour family rather than a rainbow,
 * while still being far enough apart to tell rows apart at a glance.
 *
 * The window tracks the brand: at #105915 the brand sits at 124° of the wheel,
 * so the spread runs from leaf green through the brand hue to spring green.
 * Saturation and lightness come from --brand and stay fixed -- only the hue
 * moves, which is what keeps the set coherent.
 *
 * The value is derived from the series name so adding a series needs no
 * token here, and the same name always yields the same colour across builds.
 */
export function seriesHue(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) >>> 0;
  }
  return 94 + (h % 7) * 10; // 94 .. 154
}

/** Accent for a post: its series if it has one, otherwise the brand hue. */
export function postHue(series?: string): number {
  return series ? seriesHue(series) : 124;
}