/**
 * Python ``str.casefold()`` and ``re.findall(r"\w+|[^\w\s]", ...)`` equivalents
 * used by vocabulary tokenizers. JavaScript lacks full case folding; the
 * special (multi-character) Unicode CaseFolding "F" mappings are applied after
 * ``toLowerCase()``.
 */

const SPECIAL_FOLDS: Record<string, string> = {
  'ß': 'ss', 'ẞ': 'ss', 'ſ': 's', 'ς': 'σ', 'µ': 'μ', 'ŉ': 'ʼn', 'ǰ': 'ǰ', 'ΐ': 'ΐ', 'ΰ': 'ΰ', 'և': 'եւ',
  'ẖ': 'ẖ', 'ẗ': 'ẗ', 'ẘ': 'ẘ', 'ẙ': 'ẙ', 'ẚ': 'aʾ', 'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
  'ﬅ': 'st', 'ﬆ': 'st', 'ﬓ': 'մն', 'ﬔ': 'մե', 'ﬕ': 'մի', 'ﬖ': 'վն', 'ﬗ': 'մխ', 'ϐ': 'β', 'ϑ': 'θ', 'ϕ': 'φ',
  'ϖ': 'π', 'ϰ': 'κ', 'ϱ': 'ρ', 'ϵ': 'ε', 'ᲀ': 'в', 'ᲁ': 'д', 'ᲂ': 'о', 'ᲃ': 'с', 'ᲄ': 'т', 'ᲅ': 'т', 'ᲆ': 'ъ',
  'ᲇ': 'ѣ', 'ᲈ': 'ꙋ', 'ẛ': 'ṡ', 'ι': 'ι',
};

/** Python ``str.casefold()`` (full case folding for the common special cases). */
export function casefold(text: string): string {
  let result = '';
  for (const char of text.toLowerCase()) result += SPECIAL_FOLDS[char] ?? char;
  return result;
}

/** Python ``str.lower()``. */
export function lower(text: string): string {
  return text.toLowerCase();
}

// Python's ``\w`` is ``str.isalnum()`` (letters and numbers) or underscore.
const WORDS = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;

/** Python ``re.findall(r"\w+|[^\w\s]", text)`` for Unicode strings. */
export function wordTokens(text: string): string[] {
  return text.match(WORDS) ?? [];
}
