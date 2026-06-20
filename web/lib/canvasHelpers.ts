import { CanvasItem, HandwritingPath } from "./types";

export type ParsedHandwriting = {
  paths: HandwritingPath[];
  viewBox: string;
  width: number;
  height: number;
};

export const SMALL_KANA_CHARS = new Set(
  Array.from("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"),
);
export const SMALL_YOON_CHARS = new Set(Array.from("ゃゅょャュョ"));
export const SMALL_SOKUON_CHARS = new Set(Array.from("っッ"));
export const KATAKANA_CHARS = new Set(Array.from("ァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"));
export const KANA_CHAR_RE = /[\u3041-\u3096\u30A1-\u30FA]/;
export const KATAKANA_CHAR_RE = /[\u30A1-\u30FA]/;
export const SMALL_PUNCT_CHARS = new Set(Array.from("。、，．.,・"));
export const MEDIUM_PUNCT_CHARS = new Set(Array.from("：；:;…"));
export const TALL_PUNCT_CHARS = new Set(Array.from("！？!?"));
export const BRACKET_PUNCT_CHARS = new Set(Array.from("「」『』（）()【】[]［］{}｛｝〈〉《》〔〕"));
export const DASH_PUNCT_CHARS = new Set(Array.from("ー〜—-－"));
export const OPERATOR_PUNCT_CHARS = new Set(Array.from("／/＼\\｜|＋+=＝＊*＆&％%＃#＠@￥¥$"));
export const QUOTE_PUNCT_CHARS = new Set(Array.from("“”‘’\"'"));
export const PUNCT_CHARS = new Set([
  ...SMALL_PUNCT_CHARS,
  ...MEDIUM_PUNCT_CHARS,
  ...TALL_PUNCT_CHARS,
  ...BRACKET_PUNCT_CHARS,
  ...DASH_PUNCT_CHARS,
  ...OPERATOR_PUNCT_CHARS,
  ...QUOTE_PUNCT_CHARS,
]);

export function isSmallKana(ch: string): boolean {
  return ch.length === 1 && SMALL_KANA_CHARS.has(ch);
}

export function isFullSizeKana(ch: string): boolean {
  return ch.length === 1 && KANA_CHAR_RE.test(ch) && !SMALL_KANA_CHARS.has(ch);
}

export function isSmallKatakana(ch: string): boolean {
  return ch.length === 1 && KATAKANA_CHARS.has(ch);
}

export function isPunctuation(ch: string): boolean {
  return ch.length === 1 && PUNCT_CHARS.has(ch);
}

export function isPunctuationOnlyText(text: string): boolean {
  const chars = Array.from(text.trim()).filter((ch) => ch.trim() !== "");
  return chars.length > 0 && chars.every((ch) => isPunctuation(ch));
}

export function isAsciiRenderableText(text: string): boolean {
  const chars = Array.from(text.trim()).filter((ch) => ch.trim() !== "");
  return chars.length > 0 && chars.every((ch) => {
    const code = ch.charCodeAt(0);
    return code >= 0x20 && code <= 0x7e;
  });
}

export function isGeneratedPunctuationItem(item: CanvasItem): boolean {
  return item.type === "text" && item.id.includes("_text_punct_") && isPunctuationOnlyText(item.text);
}

export function isFullSizeKatakana(ch: string): boolean {
  return ch.length === 1 && KATAKANA_CHAR_RE.test(ch) && !SMALL_KANA_CHARS.has(ch);
}

export function smallKanaScale(ch: string): number {
  if (SMALL_YOON_CHARS.has(ch)) {
    return 0.62;
  }
  return 0.5;
}

export function isLatinChar(ch: string): boolean {
  if (ch.length !== 1) return false;
  return /^[A-Za-z0-9]$/.test(ch);
}

export function getLatinCharLayout(ch: string, targetH: number, targetW: number) {
  if (ch.length !== 1) return { boxW: targetW, boxH: targetH, shiftX: 0, shiftY: 0, advanceW: targetW };

  const isUpper = /^[A-Z0-9]$/.test(ch);
  const isTallLower = /^[bdfhklt]$/.test(ch);
  const isShortLower = /^[acemnorsuvwxz]$/.test(ch);
  const isDescenderLower = /^[gpqy]$/.test(ch);

  let scale = 1.0;
  let shiftY = 0.0;
  let shiftX = 0.0;

  if (isUpper) {
    scale = 1.0;
    shiftY = 0.0;
  } else if (isTallLower) {
    scale = 0.92;
    shiftY = targetH * 0.08;
  } else if (isShortLower) {
    scale = 0.65;
    shiftY = targetH * 0.35;
  } else if (isDescenderLower) {
    scale = 0.88;
    shiftY = targetH * 0.32;
  } else if (ch === 'i') {
    scale = 0.82;
    shiftY = targetH * 0.18;
  } else if (ch === 'j') {
    scale = 1.0;
    shiftY = targetH * 0.18;
  }

  const boxH = targetH * scale;
  const boxW = targetW * scale;
  const advanceW = boxW;

  return { boxW, boxH, shiftX, shiftY, advanceW };
}

export function buildPunctuationPaths(ch: string): ParsedHandwriting | null {
  const mk = (paths: Array<[string, number]>, viewBox: string, width: number, height: number): ParsedHandwriting => ({
    paths: paths.map(([d, strokeWidth], idx) => ({
      id: `punct_${ch}_${idx}`,
      d,
      strokeWidth,
    })),
    viewBox,
    width,
    height,
  });

  if (ch === "。" || ch === "．" || ch === ".") {
    return mk(
      [["M6.4 4.0 C6.2 5.7 5.1 6.6 3.7 6.4 C2.1 6.2 1.3 5.0 1.6 3.6 C1.9 2.1 3.1 1.3 4.6 1.6 C5.8 1.8 6.6 2.8 6.4 4.0", 1.18]],
      "0 0 8 8",
      8,
      8,
    );
  }
  if (ch === "、" || ch === "，" || ch === ",") {
    return mk(
      [["M3.9 3.4 C4.2 6.0 5.0 8.5 6.5 10.7", 1.45]],
      "0 0 9 12",
      9,
      12,
    );
  }
  if (ch === "・") {
    return mk(
      [["M4.7 3.9 C4.7 5.1 3.7 5.8 2.7 5.5 C1.7 5.2 1.4 4.0 2.1 3.2 C2.9 2.3 4.3 2.7 4.7 3.9", 1.05]],
      "0 0 6 7",
      6,
      7,
    );
  }
  if (ch === "！" || ch === "!") {
    return mk(
      [
        ["M9 2 C8.8 18 8.0 34 7.0 50", 2],
        ["M7.6 61 C10.0 60.8 11.2 63.2 9.6 65.0 C7.8 66.8 5.5 65.4 6.0 63.3 C6.2 62.1 6.8 61.4 7.6 61", 1.5],
      ],
      "0 0 18 68",
      18,
      68,
    );
  }
  if (ch === "？" || ch === "?") {
    return mk(
      [
        ["M7 18 C10 6 29 6 29 20 C29 31 17 32 17 45", 2],
        ["M17 61 C19.6 60.8 20.9 63.2 19.1 65.1 C17.1 67.1 14.8 65.4 15.4 63.1 C15.6 62.0 16.2 61.3 17 61", 1.5],
      ],
      "0 0 36 68",
      36,
      68,
    );
  }
  if (ch === "ー" || ch === "〜") {
    const path = ch === "〜" ? "M3 11 C12 4 22 18 34 11" : "M3 10 C13 11 24 10 35 10";
    return mk([[path, 2]], "0 0 38 22", 38, 22);
  }
  if (ch === "…" || ch === "：" || ch === ":" || ch === "；" || ch === ";") {
    const dots = ch === "…" ? [[8, 14], [20, 14], [32, 14]] : [[20, 8], [20, 28]];
    const paths: Array<[string, number]> = dots.map(([x, y]) => [
      `M${x + 2} ${y} C${x + 2} ${y + 2.2} ${x - 1.5} ${y + 2.3} ${x - 2} ${y} C${x - 2.2} ${y - 2.0} ${x + 1.5} ${y - 2.2} ${x + 2} ${y}`,
      1.4,
    ]);
    if (ch === "；" || ch === ";") {
      paths.push(["M22 28 C20 32 18 34 15 36", 1.2]);
    }
    return mk(paths, "0 0 40 38", 40, 38);
  }
  if (BRACKET_PUNCT_CHARS.has(ch)) {
    const left = ch === "「" || ch === "『" || ch === "（" || ch === "(";
    const round = ch === "（" || ch === "）" || ch === "(" || ch === ")";
    if (round) {
      const d = left ? "M22 4 C8 18 8 48 22 64" : "M6 4 C20 18 20 48 6 64";
      return mk([[d, 1.8]], "0 0 28 68", 28, 68);
    }
    const d = left ? "M24 5 L6 5 L6 64" : "M5 64 L23 64 L23 5";
    return mk([[d, 1.8]], "0 0 28 68", 28, 68);
  }
  return null;
}

export function buildPunctuationSvg(ch: string, viewBox: string, paths: HandwritingPath[]): string {
  const pathSvg = paths
    .map(
      (p) =>
        `<path d="${p.d}" fill="none" stroke="currentColor" stroke-width="${p.strokeWidth}" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>`,
    )
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">${pathSvg}</svg>`;
}

export function handwritingBoxSize(item: CanvasItem): { width: number; height: number } {
  const width = item.boxWidth ?? 80;
  const height = item.boxHeight ?? 56;
  if (!isPunctuation(item.text)) {
    return { width, height };
  }
  if (item.type === "handwriting") {
    if (SMALL_PUNCT_CHARS.has(item.text)) {
      if (item.text === "、" || item.text === "，" || item.text === ",") {
        return { width: Math.max(12, width), height: Math.max(20, height) };
      }
      return { width: Math.max(18, width), height: Math.max(18, height) };
    }
    return { width, height };
  }
  const basis = Math.max(width, height);
  if (SMALL_PUNCT_CHARS.has(item.text)) {
    if (item.text === "、" || item.text === "，" || item.text === ",") {
      const h = Math.max(5, Math.min(11, basis * 0.2));
      return { width: Math.max(4, h * 0.55), height: h };
    }
    const h = Math.max(4, Math.min(8, basis * 0.14));
    return { width: h, height: h };
  }
  if (MEDIUM_PUNCT_CHARS.has(item.text)) {
    const h = Math.max(8, Math.min(14, basis * 0.26));
    return { width: Math.max(8, h * 1.15), height: h };
  }
  if (DASH_PUNCT_CHARS.has(item.text)) {
    const h = Math.max(5, Math.min(10, basis * 0.18));
    return { width: Math.max(18, basis * 0.58), height: h };
  }
  if (TALL_PUNCT_CHARS.has(item.text)) {
    const h = Math.max(18, Math.min(34, basis * 0.72));
    return { width: Math.max(8, h * 0.28), height: h };
  }
  if (BRACKET_PUNCT_CHARS.has(item.text)) {
    const h = Math.max(18, Math.min(36, basis * 0.78));
    return { width: Math.max(8, h * 0.28), height: h };
  }
  return { width, height };
}

export function punctuationRenderItem(item: CanvasItem): CanvasItem {
  if (
    item.type !== "handwriting" ||
    !isPunctuation(item.text) ||
    (item.paths?.length ?? 0) > 0 ||
    Boolean(item.svg)
  ) {
    return item;
  }
  const draft = buildPunctuationPaths(item.text);
  const size = handwritingBoxSize(item);
  return {
    ...item,
    type: "handwriting",
    svg: draft ? buildPunctuationSvg(item.text, draft.viewBox, draft.paths) : "",
    paths: draft?.paths ?? [],
    svgViewBox: draft?.viewBox ?? item.svgViewBox ?? "0 0 10 10",
    boxWidth: size.width,
    boxHeight: size.height,
    isConverted: true,
  };
}

export function parseSvgDimensions(svg: string): { width: number; height: number; viewBox: string } | null {
  try {
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const root = doc.documentElement;
    const vb = root.getAttribute("viewBox");
    if (vb) {
      const nums = vb
        .split(/[ ,]+/)
        .map((s) => Number(s))
        .filter((n) => Number.isFinite(n));
      if (nums.length === 4) {
        const w = Math.max(1, nums[2]);
        const h = Math.max(1, nums[3]);
        return { width: w, height: h, viewBox: `${nums[0]} ${nums[1]} ${w} ${h}` };
      }
    }
    const wAttr = root.getAttribute("width");
    const hAttr = root.getAttribute("height");
    const w = wAttr ? Number(String(wAttr).replace(/[^\d.\\-]/g, "")) : NaN;
    const h = hAttr ? Number(String(hAttr).replace(/[^\d.\\-]/g, "")) : NaN;
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      return { width: w, height: h, viewBox: `0 0 ${w} ${h}` };
    }
    return null;
  } catch {
    return null;
  }
}

export function pathBoundsFromD(d: string): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const nums = d.match(/-?\d*\.?\d+/g);
  if (!nums || nums.length < 2) return null;
  const vals = nums.map((s) => Number(s)).filter((n) => Number.isFinite(n));
  if (vals.length < 2) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 1 < vals.length; i += 2) {
    const x = vals[i];
    const y = vals[i + 1];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null;
  }
  return { minX, minY, maxX, maxY };
}

export function parseHandwritingSvg(svg: string, isJapanese: boolean): ParsedHandwriting | null {
  try {
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const root = doc.documentElement;
    let pathEls = Array.from(root.querySelectorAll("path"));
    if (pathEls.length === 0) {
      const fallbackEls: Array<{ d: string; strokeWidth: number; strokeOpacity?: number }> = [];
      const pathRe = /<path\b[^>]*\bd=['"]([^'"]+)['"][^>]*>/gi;
      let m: RegExpExecArray | null;
      while ((m = pathRe.exec(svg)) !== null) {
        const full = m[0] ?? "";
        const d = m[1] ?? "";
        const swMatch = full.match(/\bstroke-width=['"]([^'"]+)['"]/i);
        const sw = swMatch ? Number(swMatch[1]) : 1;
        const soMatch = full.match(/\bstroke-opacity=['"]([^'"]+)['"]/i);
        const so = soMatch ? Number(soMatch[1]) : undefined;
        if (d.trim()) {
          fallbackEls.push({ 
            d, 
            strokeWidth: Number.isFinite(sw) ? sw : 1,
            strokeOpacity: so !== undefined && Number.isFinite(so) ? so : undefined
          });
        }
      }
      if (fallbackEls.length === 0) return null;

      const paths: HandwritingPath[] = [];
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      fallbackEls.forEach((it, idx) => {
        paths.push({
          id: `pf_${idx}_${Math.random().toString(36).slice(2, 8)}`,
          d: it.d,
          strokeWidth: it.strokeWidth,
          strokeOpacity: it.strokeOpacity,
        });
        const b = pathBoundsFromD(it.d);
        if (b) {
          minX = Math.min(minX, b.minX);
          minY = Math.min(minY, b.minY);
          maxX = Math.max(maxX, b.maxX);
          maxY = Math.max(maxY, b.maxY);
        }
      });
      if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
        minX = 0;
        minY = 0;
        maxX = 100;
        maxY = 100;
      }
      const pad = 2;
      let vbW = Math.max(8, maxX - minX + pad * 2);
      let vbH = Math.max(8, maxY - minY + pad * 2);
      let vbX = minX - pad;
      let vbY = minY - pad;

      if (isJapanese) {
        const vbSize = Math.max(vbW, vbH);
        vbX = minX - pad - (vbSize - vbW) / 2;
        vbY = minY - pad - (vbSize - vbH) / 2;
        vbW = vbSize;
        vbH = vbSize;
      }

      return {
        paths,
        viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
        width: vbW,
        height: vbH,
      };
    }

    const paths: HandwritingPath[] = [];
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    pathEls.forEach((el, idx) => {
      const d = el.getAttribute("d") ?? "";
      if (!d.trim()) return;
      const strokeW = Number(el.getAttribute("stroke-width") ?? "1");
      const strokeO = el.getAttribute("stroke-opacity");
      paths.push({
        id: `p_${idx}_${Math.random().toString(36).slice(2, 8)}`,
        d,
        strokeWidth: Number.isFinite(strokeW) ? strokeW : 1,
        strokeOpacity: strokeO ? Number(strokeO) : undefined,
      });
      const b = pathBoundsFromD(d);
      if (b) {
        minX = Math.min(minX, b.minX);
        minY = Math.min(minY, b.minY);
        maxX = Math.max(maxX, b.maxX);
        maxY = Math.max(maxY, b.maxY);
      }
    });
    if (paths.length === 0) return null;
    if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
      minX = 0;
      minY = 0;
      maxX = 100;
      maxY = 100;
    }
    const pad = 2;
    let vbW = Math.max(8, maxX - minX + pad * 2);
    let vbH = Math.max(8, maxY - minY + pad * 2);
    let vbX = minX - pad;
    let vbY = minY - pad;

    if (isJapanese) {
      const vbSize = Math.max(vbW, vbH);
      vbX = minX - pad - (vbSize - vbW) / 2;
      vbY = minY - pad - (vbSize - vbH) / 2;
      vbW = vbSize;
      vbH = vbSize;
    }

    return {
      paths,
      viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
      width: vbW,
      height: vbH,
    };
  } catch {
    return null;
  }
}

export const rotatePoint = (angleDeg: number, point: { x: number; y: number }): { x: number; y: number } => {
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return {
    x: point.x * cos - point.y * sin,
    y: point.x * sin + point.y * cos,
  };
};

export const getItemSize = (item: CanvasItem) => {
  if (item.type === "text") {
    return {
      width: item.boxWidth ?? 400,
      height: item.boxHeight ?? 200,
    };
  } else if (item.type === "handwriting") {
    const size = handwritingBoxSize(item);
    return {
      width: item.boxWidth ?? size.width,
      height: item.boxHeight ?? size.height,
    };
  } else {
    // shapes
    return {
      width: item.boxWidth ?? 120,
      height: item.boxHeight ?? 120,
    };
  }
};
