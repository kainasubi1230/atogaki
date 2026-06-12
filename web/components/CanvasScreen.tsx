import { useEffect, useState, useCallback } from "react";
import { CanvasItem, HandwritingPath } from "../lib/types";
import { COLORS, PAPERS } from "../lib/constants";
import { getJSON, postJSON } from "../lib/api";
import { ShapeRenderer } from "./ShapeRenderer";

type Props = {
  token: string;
  userId: number | null;
  styleId: number | null;
};

type StyleCoverage = {
  style_id: number;
  status: string;
  total_target_chars: number;
  covered_count: number;
  missing_count: number;
  covered_chars: string;
  missing_hiragana: string;
  missing_katakana: string;
  missing_kanji_core: string;
};

type ParsedHandwriting = {
  paths: HandwritingPath[];
  viewBox: string;
  width: number;
  height: number;
};

const SMALL_KANA_CHARS = new Set(
  Array.from("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"),
);
const SMALL_YOON_CHARS = new Set(Array.from("ゃゅょャュョ"));
const SMALL_SOKUON_CHARS = new Set(Array.from("っッ"));
const KATAKANA_CHARS = new Set(Array.from("ァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"));
const KANA_CHAR_RE = /[\u3041-\u3096\u30A1-\u30FA]/;
const KATAKANA_CHAR_RE = /[\u30A1-\u30FA]/;
const SMALL_PUNCT_CHARS = new Set(Array.from("。、，．.,・"));
const MEDIUM_PUNCT_CHARS = new Set(Array.from("：；:;…"));
const TALL_PUNCT_CHARS = new Set(Array.from("！？!?"));
const BRACKET_PUNCT_CHARS = new Set(Array.from("「」『』（）()【】[]［］{}｛｝〈〉《》〔〕"));
const DASH_PUNCT_CHARS = new Set(Array.from("ー〜—-－"));
const OPERATOR_PUNCT_CHARS = new Set(Array.from("／/＼\\｜|＋+=＝＊*＆&％%＃#＠@￥¥$"));
const QUOTE_PUNCT_CHARS = new Set(Array.from("“”‘’\"'"));
const PUNCT_CHARS = new Set([
  ...SMALL_PUNCT_CHARS,
  ...MEDIUM_PUNCT_CHARS,
  ...TALL_PUNCT_CHARS,
  ...BRACKET_PUNCT_CHARS,
  ...DASH_PUNCT_CHARS,
  ...OPERATOR_PUNCT_CHARS,
  ...QUOTE_PUNCT_CHARS,
]);

function isSmallKana(ch: string): boolean {
  return ch.length === 1 && SMALL_KANA_CHARS.has(ch);
}

function isFullSizeKana(ch: string): boolean {
  return ch.length === 1 && KANA_CHAR_RE.test(ch) && !SMALL_KANA_CHARS.has(ch);
}

function isSmallKatakana(ch: string): boolean {
  return ch.length === 1 && KATAKANA_CHARS.has(ch);
}

function isPunctuation(ch: string): boolean {
  return ch.length === 1 && PUNCT_CHARS.has(ch);
}

function isPunctuationOnlyText(text: string): boolean {
  const chars = Array.from(text.trim()).filter((ch) => ch.trim() !== "");
  return chars.length > 0 && chars.every((ch) => isPunctuation(ch));
}

function isAsciiRenderableText(text: string): boolean {
  const chars = Array.from(text.trim()).filter((ch) => ch.trim() !== "");
  return chars.length > 0 && chars.every((ch) => {
    const code = ch.charCodeAt(0);
    return code >= 0x20 && code <= 0x7e;
  });
}

function isGeneratedPunctuationItem(item: CanvasItem): boolean {
  return item.type === "text" && item.id.includes("_text_punct_") && isPunctuationOnlyText(item.text);
}

function isFullSizeKatakana(ch: string): boolean {
  return ch.length === 1 && KATAKANA_CHAR_RE.test(ch) && !SMALL_KANA_CHARS.has(ch);
}

function smallKanaScale(ch: string): number {
  if (SMALL_YOON_CHARS.has(ch)) {
    return 0.62;
  }
  return 0.5;
}

function buildPunctuationPaths(ch: string): ParsedHandwriting | null {
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

function buildPunctuationSvg(ch: string, viewBox: string, paths: HandwritingPath[]): string {
  const pathSvg = paths
    .map(
      (p) =>
        `<path d="${p.d}" fill="none" stroke="currentColor" stroke-width="${p.strokeWidth}" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>`,
    )
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">${pathSvg}</svg>`;
}

function PunctuationMark({ ch, color }: { ch: string; color: string }) {
  if (ch === "。" || ch === "．" || ch === ".") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 18 18"
        style={{
          position: "absolute",
          left: "16%",
          top: "24%",
          width: "62%",
          height: "62%",
          overflow: "visible",
          pointerEvents: "none",
        }}
      >
        <path
          d="M12.4 7.5 C12.0 10.6 9.8 12.7 7.0 12.1 C4.6 11.6 3.5 9.5 4.1 7.1 C4.8 4.7 7.1 3.6 9.4 4.2 C11.4 4.7 12.7 5.9 12.4 7.5"
          fill="none"
          stroke={color}
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (ch === "・") {
    return (
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          left: "35%",
          top: "35%",
          width: "30%",
          height: "30%",
          border: `1.4px solid ${color}`,
          borderRadius: "50%",
          boxSizing: "border-box",
          display: "block",
          pointerEvents: "none",
        }}
      />
    );
  }
  if (ch === "、" || ch === "，" || ch === ",") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 18 28"
        style={{
          position: "absolute",
          left: "26%",
          top: "5%",
          width: "58%",
          height: "86%",
          overflow: "visible",
          pointerEvents: "none",
        }}
      >
        <path
          d="M5.0 4.2 C6.4 10.5 8.8 17.2 13.5 23.2"
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (ch === "！" || ch === "!") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 22 58"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}
      >
        <path d="M12.2 4.5 C11.4 17.0 10.8 29.5 9.4 41.0" fill="none" stroke={color} strokeWidth="2.1" strokeLinecap="round" />
        <path d="M9.5 50.5 C11.8 50.1 13.0 52.4 11.2 54.0 C9.4 55.5 7.2 54.1 7.7 52.1 C8.0 51.0 8.6 50.6 9.5 50.5" fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (ch === "？" || ch === "?") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 34 58"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}
      >
        <path d="M7.2 16.5 C8.8 6.2 25.4 4.7 27.0 16.6 C28.3 26.0 17.3 27.1 16.1 37.4" fill="none" stroke={color} strokeWidth="2.0" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M15.5 50.2 C18.0 49.8 19.1 52.1 17.3 53.9 C15.6 55.5 13.1 54.2 13.7 52.0 C14.0 50.9 14.6 50.4 15.5 50.2" fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return null;
}

function handwritingBoxSize(item: CanvasItem): { width: number; height: number } {
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

function punctuationRenderItem(item: CanvasItem): CanvasItem {
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

function parseSvgDimensions(svg: string): { width: number; height: number; viewBox: string } | null {
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

function pathBoundsFromD(d: string): { minX: number; minY: number; maxX: number; maxY: number } | null {
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

function parseHandwritingSvg(svg: string): ParsedHandwriting | null {
  try {
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const root = doc.documentElement;
    let pathEls = Array.from(root.querySelectorAll("path"));
    if (pathEls.length === 0) {
      const fallbackEls: Array<{ d: string; strokeWidth: number }> = [];
      const pathRe = /<path\b[^>]*\bd=['"]([^'"]+)['"][^>]*>/gi;
      let m: RegExpExecArray | null;
      while ((m = pathRe.exec(svg)) !== null) {
        const full = m[0] ?? "";
        const d = m[1] ?? "";
        const swMatch = full.match(/\bstroke-width=['"]([^'"]+)['"]/i);
        const sw = swMatch ? Number(swMatch[1]) : 1;
        if (d.trim()) {
          fallbackEls.push({ d, strokeWidth: Number.isFinite(sw) ? sw : 1 });
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
      const vbX = minX - pad;
      const vbY = minY - pad;
      const vbW = Math.max(8, maxX - minX + pad * 2);
      const vbH = Math.max(8, maxY - minY + pad * 2);
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
      paths.push({
        id: `p_${idx}_${Math.random().toString(36).slice(2, 8)}`,
        d,
        strokeWidth: Number.isFinite(strokeW) ? strokeW : 1,
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
    const vbX = minX - pad;
    const vbY = minY - pad;
    const vbW = Math.max(8, maxX - minX + pad * 2);
    const vbH = Math.max(8, maxY - minY + pad * 2);
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

export function CanvasScreen({ token, userId, styleId }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(true);
  const [coverage, setCoverage] = useState<StyleCoverage | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(false);

  const [textColor, setTextColor] = useState(COLORS[0].color);
  const [textSize, setTextSize] = useState(24);
  const [paperStyle, setPaperStyle] = useState(PAPERS[0].id);
  const [orientation, setOrientation] = useState<"portrait" | "landscape">(
    "portrait",
  );

  const [historyState, setHistoryState] = useState({
    past: [] as CanvasItem[][],
    present: [
      {
        id: "1",
        type: "text",
        x: 40,
        y: 40,
        text: "",
        svg: "",
        color: COLORS[0].color,
        isConverted: false,
        fontSize: 24,
        letterSpacing: 0,
        lineHeight: 1.5,
      },
    ] as CanvasItem[],
    future: [] as CanvasItem[][],
  });

  const items = historyState.present;

  const setItems = (
    newItemsOrUpdater: CanvasItem[] | ((prev: CanvasItem[]) => CanvasItem[]),
  ) => {
    setHistoryState((prev) => {
      const nextPresent =
        typeof newItemsOrUpdater === "function"
          ? newItemsOrUpdater(prev.present)
          : newItemsOrUpdater;
      return { ...prev, present: nextPresent };
    });
  };

  const commitHistory = useCallback(() => {
    setHistoryState((prev) => {
      if (
        prev.past.length > 0 &&
        JSON.stringify(prev.past[prev.past.length - 1]) ===
          JSON.stringify(prev.present)
      ) {
        return prev;
      }
      return {
        past: [...prev.past, prev.present].slice(-50),
        present: prev.present,
        future: [],
      };
    });
  }, []);

  // Global Undo/Redo Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "z") {
        if (
          document.activeElement?.tagName === "TEXTAREA" ||
          document.activeElement?.tagName === "INPUT"
        )
          return;
        e.preventDefault();
        if (e.shiftKey) {
          setHistoryState((prev) => {
            if (prev.future.length === 0) return prev;
            return {
              past: [...prev.past, prev.present],
              present: prev.future[0],
              future: prev.future.slice(1),
            };
          });
        } else {
          setHistoryState((prev) => {
            if (prev.past.length === 0) return prev;
            return {
              past: prev.past.slice(0, -1),
              present: prev.past[prev.past.length - 1],
              future: [prev.present, ...prev.future],
            };
          });
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const [selectedId, setSelectedId] = useState<string | null>("1");

  // Drag logic states
  const [isDragging, setIsDragging] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [activeTool, setActiveTool] = useState<
    "pointer" | "text" | "square" | "circle" | "triangle" | "eraser"
  >(
    "pointer",
  );

  const [textLetterSpacing, setTextLetterSpacing] = useState(0);
  const [textLineHeight, setTextLineHeight] = useState(1.5);

  useEffect(() => {
    const handleGlobalMove = (e: PointerEvent) => {
      if (isDragging) {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        setItems((prev) =>
          prev.map((it) =>
            it.id === isDragging ? { ...it, x: it.x + dx, y: it.y + dy } : it,
          ),
        );
        setDragStart({ x: e.clientX, y: e.clientY });
      }
    };
    const handleGlobalUp = () => {
      setIsDragging((prev) => {
        if (prev !== null) commitHistory();
        return null;
      });
    };

    if (isDragging) {
      window.addEventListener("pointermove", handleGlobalMove);
      window.addEventListener("pointerup", handleGlobalUp);
    }
    return () => {
      window.removeEventListener("pointermove", handleGlobalMove);
      window.removeEventListener("pointerup", handleGlobalUp);
    };
  }, [isDragging, dragStart, commitHistory]);

  const handlePointerDown = (e: React.PointerEvent, id: string) => {
    setIsDragging(id);
    setSelectedId(id);
    setDragStart({ x: e.clientX, y: e.clientY });
    e.preventDefault(); // prevent text selection while dragging
  };

  const handlePaperClick = (e: React.PointerEvent) => {
    if (e.target === e.currentTarget) {
      const canCreate =
        activeTool === "text" ||
        activeTool === "square" ||
        activeTool === "circle" ||
        activeTool === "triangle";
      if (canCreate) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const newItem: CanvasItem = {
          id: Date.now().toString(),
          type: activeTool,
          x,
          y,
          text: "",
          svg: "",
          color: textColor,
          isConverted: false,
          fontSize: activeTool === "text" ? textSize : undefined,
          letterSpacing: activeTool === "text" ? textLetterSpacing : undefined,
          lineHeight: activeTool === "text" ? textLineHeight : undefined,
        };
        setItems((prev) => [...prev, newItem]);
        setSelectedId(newItem.id);
        commitHistory();

        setActiveTool("pointer");
      } else {
        setSelectedId(null);
      }
    }
  };

  const deleteItem = (id: string) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
    if (selectedId === id) setSelectedId(null);
    commitHistory();
  };

  const selectedItem = items.find((it) => it.id === selectedId);
  const currentColor = selectedItem ? selectedItem.color : textColor;
  const currentFontSize =
    selectedItem &&
    selectedItem.type === "text" &&
    selectedItem.fontSize !== undefined
      ? selectedItem.fontSize
      : textSize;
  const currentLetterSpacing =
    selectedItem &&
    selectedItem.type === "text" &&
    selectedItem.letterSpacing !== undefined
      ? selectedItem.letterSpacing
      : textLetterSpacing;
  const currentLineHeight =
    selectedItem &&
    selectedItem.type === "text" &&
    selectedItem.lineHeight !== undefined
      ? selectedItem.lineHeight
      : textLineHeight;

  const handleColorChange = (newColor: string) => {
    setTextColor(newColor);
    if (selectedId) {
      setItems(
        items.map((it) =>
          it.id === selectedId ? { ...it, color: newColor } : it,
        ),
      );
    }
  };

  const handleFontSizeChange = (newSize: number) => {
    setTextSize(newSize);
    if (selectedId) {
      setItems(
        items.map((it) =>
          it.id === selectedId && it.type === "text"
            ? { ...it, fontSize: newSize }
            : it,
        ),
      );
    }
  };

  const handleLetterSpacingChange = (newVal: number) => {
    setTextLetterSpacing(newVal);
    if (selectedId) {
      setItems(
        items.map((it) =>
          it.id === selectedId && it.type === "text"
            ? { ...it, letterSpacing: newVal }
            : it,
        ),
      );
    }
  };

  const handleLineHeightChange = (newVal: number) => {
    setTextLineHeight(newVal);
    if (selectedId) {
      setItems(
        items.map((it) =>
          it.id === selectedId && it.type === "text"
            ? { ...it, lineHeight: newVal }
            : it,
        ),
      );
    }
  };

  async function handleConvert() {
    const textsToConvert = items.filter(
      (it) =>
        it.type === "text" &&
        it.text.trim() !== "" &&
        !it.isConverted,
    );
    if (textsToConvert.length === 0) return;

    setIsProcessing(true);
    try {
      const canConvertWithoutStyle = textsToConvert.every((item) => isAsciiRenderableText(item.text));
      if (!token || !userId || (!styleId && !canConvertWithoutStyle)) {
        alert("スタイル準備ができていません。先に画像アップロードをやり直してください。");
        return;
      }

      let newItems: CanvasItem[] = [];
      let convertedCount = 0;
      let failedCount = 0;
      let lastErrorDetail = "";
      const useRuntimePunctuation = true;
      for (const item of items) {
        if (
          item.type !== "text" ||
          item.text.trim() === "" ||
          item.isConverted
        ) {
          newItems.push(item);
          continue;
        }

        const lines = item.text.split("\n");
        const generatedChars: CanvasItem[] = [];
        const baseFont = item.fontSize ?? 24;
        const lineHeightPx = (item.lineHeight ?? 1.5) * baseFont;
        const letterSpace = item.letterSpacing ?? 0;
        const generationPromises = new Map<string, ReturnType<typeof postJSON>>();
        lines.forEach((line, lineIndex) => {
          Array.from(line).forEach((ch, charIndex) => {
            if (ch.trim() === "" || (isPunctuation(ch) && !useRuntimePunctuation)) {
              return;
            }
            generationPromises.set(
              `${lineIndex}:${charIndex}`,
              postJSON(
                "/generate",
                {
                  user_id: userId,
                  style_id: styleId ?? 0,
                  text: ch,
                  purpose: "accessibility",
                },
                token,
              ),
            );
          });
        });
        let currentY = item.y;
        for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
          const line = lines[lineIndex];
          const chars = Array.from(line);
          let currentX = item.x;
          let previousPlacedChar = "";
          for (let charIndex = 0; charIndex < chars.length; charIndex += 1) {
            const ch = chars[charIndex];
            if (ch.trim() === "") {
              currentX += baseFont * 0.6 + letterSpace;
              previousPlacedChar = "";
              continue;
            }
            if (isPunctuation(ch) && !useRuntimePunctuation) {
              const targetH = Math.max(22, baseFont * 1.55);
              let boxW = baseFont * 0.45;
              let boxH = baseFont * 0.45;
              let drawX = currentX + baseFont * 0.5;
              let drawY = currentY + targetH * 0.64;
              let advanceW = baseFont * 0.34;
              let punctFontSize = Math.max(12, Math.min(baseFont * 0.7, boxH));
              if (ch === "、" || ch === "，" || ch === ",") {
                boxW = Math.max(14, baseFont * 0.46);
                boxH = Math.max(22, baseFont * 0.78);
                drawX = currentX + baseFont * 0.3;
                drawY = currentY + targetH * 0.36;
                advanceW = baseFont * 0.24;
                punctFontSize = Math.max(14, Math.min(baseFont * 0.82, boxH));
              } else if (ch === "。" || ch === "．" || ch === "." || ch === "・") {
                boxW = Math.max(14, baseFont * 0.46);
                boxH = Math.max(14, baseFont * 0.46);
                drawX = currentX + baseFont * 0.18;
                drawY = currentY + targetH * 0.47;
                advanceW = baseFont * 0.18;
                punctFontSize = Math.max(12, Math.min(baseFont * 0.58, boxH));
              } else if (ch === "…" || ch === "：" || ch === ":" || ch === "；" || ch === ";") {
                boxW = baseFont * 0.55;
                boxH = baseFont * 0.34;
                drawX = currentX + baseFont * 0.2;
                drawY = currentY + targetH * 0.46;
                advanceW = baseFont * 0.58;
              } else if (ch === "ー" || ch === "〜") {
                boxW = baseFont * 0.78;
                boxH = baseFont * 0.24;
                drawX = currentX + baseFont * 0.08;
                drawY = currentY + targetH * 0.48;
                advanceW = baseFont * 0.82;
              } else if (ch === "！" || ch === "!" || ch === "？" || ch === "?") {
                boxW = baseFont * 0.36;
                boxH = targetH * 0.82;
                drawX = currentX + baseFont * 0.26;
                drawY = currentY + targetH * 0.08;
                advanceW = baseFont * 0.48;
              } else if (BRACKET_PUNCT_CHARS.has(ch)) {
                boxW = baseFont * 0.36;
                boxH = targetH * 0.9;
                drawX = currentX + baseFont * 0.12;
                drawY = currentY + targetH * 0.04;
                advanceW = baseFont * 0.48;
              }
              generatedChars.push({
                id: `${item.id}_text_punct_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                type: "text",
                x: drawX,
                y: drawY,
                text: ch,
                svg: "",
                color: item.color,
                isConverted: true,
                fontSize: punctFontSize,
                letterSpacing: 0,
                lineHeight: 1,
                paths: [],
                svgViewBox: "0 0 10 10",
                boxWidth: boxW,
                boxHeight: boxH,
              });
              convertedCount += 1;
              currentX += advanceW + Math.max(0.5, letterSpace * 0.35);
              previousPlacedChar = ch;
              continue;
            }
            const result = await generationPromises.get(`${lineIndex}:${charIndex}`);
            if (!result) {
              failedCount += 1;
              lastErrorDetail = "missing_generation_job";
              continue;
            }
            if (result.status !== 200) {
              failedCount += 1;
              const body = result.body as { detail?: string };
              lastErrorDetail = body?.detail ?? `http_${result.status}`;
              continue;
            }
            const body = result.body as { svg: string };
            if (!body.svg || body.svg.trim() === "") {
              failedCount += 1;
              lastErrorDetail = "empty_svg";
              continue;
            }
            const parsed = parseHandwritingSvg(body.svg);
            const fallbackDims = parseSvgDimensions(body.svg);
            const targetH = Math.max(22, baseFont * 1.55);
            const refW = parsed?.width ?? fallbackDims?.width ?? baseFont;
            const refH = parsed?.height ?? fallbackDims?.height ?? baseFont;
            const ratio = refW / Math.max(1, refH);
            let targetW = Math.max(18, Math.min(220, targetH * ratio));
            let boxH = targetH;
            let drawX = currentX;
            let drawY = currentY;
            let advanceW = targetW;
            let afterGap = Math.max(2, letterSpace);
            if (isSmallKana(ch)) {
              // Small kana should occupy a quarter of the normal character box.
              const scale = smallKanaScale(ch);
              targetW *= scale;
              const smallH = targetH * scale;
              advanceW = targetW;
              let shiftX = targetW * 0.22;
              let shiftY = smallH * 0.28;
              const smallKatakana = isSmallKatakana(ch);

              if (previousPlacedChar && isFullSizeKana(previousPlacedChar)) {
                const prevFullKatakana = isFullSizeKatakana(previousPlacedChar);
                if (SMALL_YOON_CHARS.has(ch)) {
                  if (smallKatakana && prevFullKatakana) {
                    shiftX = -targetW * 0.52;
                    shiftY = smallH * 0.5;
                    advanceW = targetW * 0.72;
                    afterGap = Math.max(2, letterSpace * 0.75);
                  } else {
                    shiftX = -targetW * 0.3;
                    shiftY = smallH * 0.46;
                    advanceW = targetW * 0.7;
                  }
                } else if (SMALL_SOKUON_CHARS.has(ch)) {
                  if (smallKatakana && prevFullKatakana) {
                    shiftX = -targetW * 0.42;
                    shiftY = smallH * 0.48;
                    advanceW = targetW * 0.68;
                    afterGap = Math.max(2, letterSpace * 0.7);
                  } else {
                    shiftX = -targetW * 0.18;
                    shiftY = smallH * 0.4;
                    advanceW = targetW * 0.72;
                  }
                } else {
                  if (smallKatakana && prevFullKatakana) {
                    shiftX = -targetW * 0.34;
                    shiftY = smallH * 0.44;
                    advanceW = targetW * 0.7;
                    afterGap = Math.max(2, letterSpace * 0.7);
                  } else {
                    shiftX = -targetW * 0.1;
                    shiftY = smallH * 0.34;
                    advanceW = targetW * 0.76;
                  }
                }
                if (smallKatakana) {
                  shiftY += smallH * 0.08;
                }
              }

              drawX += shiftX;
              drawY += shiftY;
            } else if (isPunctuation(ch)) {
              if (SMALL_PUNCT_CHARS.has(ch)) {
                const punctH = Math.max(4, baseFont * (ch === "、" || ch === "，" || ch === "," ? 0.3 : 0.22));
                targetW = Math.max(4, punctH * ratio);
                boxH = punctH;
                drawX += baseFont * (ch === "、" || ch === "，" || ch === "," ? 0.32 : 0.56);
                drawY += targetH * (ch === "、" || ch === "，" || ch === "," ? 0.62 : 0.68);
                advanceW = baseFont * (ch === "、" || ch === "，" || ch === "," ? 0.24 : 0.28);
                afterGap = Math.max(0.5, letterSpace * 0.25);
              } else if (MEDIUM_PUNCT_CHARS.has(ch)) {
                const punctH = Math.max(10, baseFont * 0.42);
                targetW = Math.max(12, Math.min(baseFont * 0.58, punctH * ratio));
                boxH = punctH;
                drawX += baseFont * 0.18;
                drawY += targetH * 0.38;
                advanceW = baseFont * 0.55;
                afterGap = Math.max(1, letterSpace * 0.45);
              } else if (DASH_PUNCT_CHARS.has(ch)) {
                boxH = Math.max(8, baseFont * 0.28);
                targetW = Math.max(baseFont * 0.62, Math.min(baseFont * 0.88, targetH * ratio));
                drawX += baseFont * 0.04;
                drawY += targetH * 0.4;
                advanceW = targetW;
              } else if (OPERATOR_PUNCT_CHARS.has(ch)) {
                boxH = targetH * 0.72;
                targetW = Math.max(baseFont * 0.35, Math.min(baseFont * 0.82, boxH * ratio));
                drawX += baseFont * 0.06;
                drawY += targetH * 0.14;
                advanceW = Math.max(baseFont * 0.48, targetW * 0.86);
              } else if (QUOTE_PUNCT_CHARS.has(ch)) {
                boxH = targetH * 0.42;
                targetW = Math.max(baseFont * 0.16, Math.min(baseFont * 0.42, boxH * ratio));
                drawX += baseFont * 0.18;
                drawY += targetH * 0.08;
                advanceW = baseFont * 0.26;
              } else if (TALL_PUNCT_CHARS.has(ch)) {
                boxH = targetH * 0.9;
                targetW = Math.max(baseFont * 0.26, Math.min(baseFont * 0.5, boxH * ratio));
                drawX += baseFont * 0.16;
                drawY += targetH * 0.04;
                advanceW = baseFont * 0.52;
              } else if (BRACKET_PUNCT_CHARS.has(ch)) {
                boxH = targetH * 0.96;
                targetW = Math.max(baseFont * 0.28, Math.min(baseFont * 0.5, boxH * ratio));
                drawX += baseFont * 0.08;
                drawY += targetH * 0.02;
                advanceW = baseFont * 0.52;
              }
            }
            generatedChars.push({
              id: `${item.id}_hw_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              type: "handwriting",
              x: drawX,
              y: drawY,
              text: ch,
              svg: body.svg,
              color: item.color,
              isConverted: true,
              paths: parsed?.paths ?? [],
              svgViewBox: parsed?.viewBox ?? fallbackDims?.viewBox ?? "0 0 100 100",
              boxWidth: targetW,
              boxHeight: isSmallKana(ch) ? targetH * smallKanaScale(ch) : boxH,
            });
            convertedCount += 1;
            currentX += advanceW + afterGap;
            previousPlacedChar = ch;
          }
          currentY += Math.max(baseFont + 2, lineHeightPx);
        }
        if (generatedChars.length > 0) {
          newItems.push(...generatedChars);
        } else {
          newItems.push(item);
        }
      }
      setItems(newItems);
      commitHistory();

      if (convertedCount === 0 && failedCount > 0) {
        const suffix = lastErrorDetail ? ` (${lastErrorDetail})` : "";
        alert(`変換に失敗しました。スタイル準備をやり直してください。${suffix}`);
      }
    } catch (e) {
      console.error(e);
      alert("変換に失敗しました");
    } finally {
      setIsProcessing(false);
    }
  }

  const hasTextToConvert = items.some(
    (it) =>
      it.type === "text" &&
      it.text.trim() !== "" &&
      !it.isConverted,
  );
  const selectedPaperDef = PAPERS.find((p) => p.id === paperStyle) || PAPERS[0];
  const customImageStyle = selectedPaperDef.image
    ? {
        backgroundImage: `url('${selectedPaperDef.image}')`,
        backgroundSize: "100% 100%",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }
    : {};

  useEffect(() => {
    async function fetchCoverage() {
      if (!token || !userId || !styleId) {
        setCoverage(null);
        return;
      }
      setCoverageLoading(true);
      try {
        const res = await getJSON(`/styles/${styleId}/coverage`, token);
        if (res.status === 200) {
          setCoverage(res.body as StyleCoverage);
        } else {
          setCoverage(null);
        }
      } finally {
        setCoverageLoading(false);
      }
    }
    fetchCoverage();
  }, [token, userId, styleId]);

  const missingGuide =
    coverage &&
    [coverage.missing_hiragana, coverage.missing_katakana, coverage.missing_kanji_core]
      .filter((s) => s && s.length > 0)
      .join("\n");

  const copyMissingGuide = async () => {
    if (!missingGuide) return;
    try {
      await navigator.clipboard.writeText(missingGuide);
      alert("不足文字をコピーしました。練習シートに貼り付けて使えます。");
    } catch {
      alert("コピーに失敗しました");
    }
  };

  return (
    <div className="canvas-screen animate-fade-in" style={{ position: "relative" }}>
      <button 
        className={`menu-toggle-btn ${!isMenuOpen ? "closed" : ""}`} 
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        title={isMenuOpen ? "メニューを閉じる" : "メニューを開く"}
      >
        {isMenuOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6"></polyline>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
        )}
      </button>

      <div className={`sidebar ${!isMenuOpen ? "closed" : ""} animate-slide-up`}>
        <div className="sidebar-section">
          <h3>便箋デザイン</h3>

          <div style={{ marginBottom: "1rem" }}>
            <div
              style={{
                fontSize: "0.85rem",
                marginBottom: "0.4rem",
                color: "var(--text-secondary)",
              }}
            >
              種類
            </div>
            <div className="paper-options">
              {PAPERS.map((p) => (
                <button
                  key={p.id}
                  className={`paper-btn ${paperStyle === p.id ? "active" : ""}`}
                  onClick={() => setPaperStyle(p.id)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div
              style={{
                fontSize: "0.85rem",
                marginBottom: "0.4rem",
                color: "var(--text-secondary)",
              }}
            >
              向き
            </div>
            <div className="tool-grid">
              <button
                className={`tool-btn ${orientation === "portrait" ? "active" : ""}`}
                onClick={() => setOrientation("portrait")}
                style={{ padding: "0.6rem 0" }}
              >
                縦向き
              </button>
              <button
                className={`tool-btn ${orientation === "landscape" ? "active" : ""}`}
                onClick={() => setOrientation("landscape")}
                style={{ padding: "0.6rem 0" }}
              >
                横向き
              </button>
            </div>
          </div>
        </div>

        <hr className="sidebar-divider" />

        <div className="sidebar-section">
          <h3>文字を書く</h3>
          <div className="tool-grid">
            <button
              className={`tool-btn ${activeTool === "pointer" ? "active" : ""}`}
              onClick={() => setActiveTool("pointer")}
              title="選択・移動"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                <path d="M13 13l6 6"></path>
              </svg>
              <span>選択</span>
            </button>
            <button
              className={`tool-btn ${activeTool === "text" ? "active" : ""}`}
              onClick={() => setActiveTool("text")}
              title="テキスト"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="4 7 4 4 20 4 20 7"></polyline>
                <line x1="9" y1="20" x2="15" y2="20"></line>
                <line x1="12" y1="4" x2="12" y2="20"></line>
              </svg>
              <span>文字</span>
            </button>
            <button
              className={`tool-btn ${activeTool === "eraser" ? "active" : ""}`}
              onClick={() => setActiveTool("eraser")}
              title="手書きの線を消す"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M20 20H7L3 16l9-9 8 8-5 5z"></path>
                <path d="M6 13l5 5"></path>
              </svg>
              <span>消しゴム</span>
            </button>
          </div>
        </div>

        <hr className="sidebar-divider" />

        <div className="sidebar-section">
          <h3>スタンプ（図形）</h3>
          <div className="tool-grid">
            <button
              className={`tool-btn ${activeTool === "square" ? "active" : ""}`}
              onClick={() => setActiveTool("square")}
              title="四角形"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              </svg>
              <span>四角</span>
            </button>
            <button
              className={`tool-btn ${activeTool === "circle" ? "active" : ""}`}
              onClick={() => setActiveTool("circle")}
              title="円形"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10"></circle>
              </svg>
              <span>丸</span>
            </button>
            <button
              className={`tool-btn ${activeTool === "triangle" ? "active" : ""}`}
              onClick={() => setActiveTool("triangle")}
              title="三角形"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              </svg>
              <span>三角</span>
            </button>
          </div>
        </div>

        <hr className="sidebar-divider" />

        <div className="sidebar-section">
          <h3>文字のスタイル</h3>

          <div>
            <div
              style={{
                fontSize: "0.85rem",
                marginBottom: "0.4rem",
                color: "var(--text-secondary)",
              }}
            >
              文字の大きさ
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span
                style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}
              >
                小
              </span>
              <input
                type="range"
                min="14"
                max="60"
                value={currentFontSize}
                onChange={(e) => handleFontSizeChange(parseInt(e.target.value))}
                onPointerUp={commitHistory}
                className="sidebar-slider"
              />
              <span
                style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}
              >
                大
              </span>
            </div>
          </div>

          <div style={{ marginTop: "0.75rem" }}>
            <div
              style={{
                fontSize: "0.85rem",
                marginBottom: "0.4rem",
                color: "var(--text-secondary)",
              }}
            >
              文字の間隔（横）
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span
                style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}
              >
                狭
              </span>
              <input
                type="range"
                min="0"
                max="20"
                step="1"
                value={currentLetterSpacing}
                onChange={(e) =>
                  handleLetterSpacingChange(parseInt(e.target.value))
                }
                onPointerUp={commitHistory}
                className="sidebar-slider"
              />
              <span
                style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}
              >
                広
              </span>
            </div>
          </div>

          <div style={{ marginTop: "0.75rem" }}>
            <div
              style={{
                fontSize: "0.85rem",
                marginBottom: "0.4rem",
                color: "var(--text-secondary)",
              }}
            >
              行の間隔（縦）
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span
                style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}
              >
                狭
              </span>
              <input
                type="range"
                min="1"
                max="4"
                step="0.1"
                value={currentLineHeight}
                onChange={(e) =>
                  handleLineHeightChange(parseFloat(e.target.value))
                }
                onPointerUp={commitHistory}
                className="sidebar-slider"
              />
              <span
                style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}
              >
                広
              </span>
            </div>
          </div>
        </div>

        <hr className="sidebar-divider" />

        <div className="sidebar-section">
          <h3>インクの色</h3>
          <div className="color-options" style={{ alignItems: "center" }}>
            {COLORS.map((c) => (
              <button
                key={c.id}
                className={`color-btn ${currentColor === c.color ? "active" : ""}`}
                style={{ backgroundColor: c.color }}
                title={c.name}
                onClick={() => {
                  handleColorChange(c.color);
                  commitHistory();
                }}
              />
            ))}
            <div className="color-picker-wrapper" title="自由な色を選ぶ">
              <input
                type="color"
                className="color-picker-input"
                value={currentColor}
                onChange={(e) => handleColorChange(e.target.value)}
                onBlur={commitHistory}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="canvas-main">
        {styleId && (
          <div
            style={{
              position: "absolute",
              top: "1.1rem",
              right: "1.5rem",
              zIndex: 101,
              maxWidth: "500px",
              background: "rgba(255,255,255,0.95)",
              border: "1px solid #e6d2bf",
              borderRadius: "12px",
              padding: "0.7rem 0.85rem",
              boxShadow: "0 6px 20px rgba(0,0,0,0.08)",
              fontSize: "0.82rem",
              lineHeight: 1.4,
            }}
          >
            {coverageLoading ? (
              <div>文字カバレッジ確認中...</div>
            ) : coverage ? (
              <>
                <div style={{ fontWeight: 700, marginBottom: "0.25rem" }}>
                  文字カバレッジ {coverage.covered_count}/{coverage.total_target_chars}
                </div>
                <div style={{ color: "var(--text-secondary)" }}>
                  不足: {coverage.missing_count} 文字
                </div>
                {coverage.missing_hiragana && (
                  <div style={{ marginTop: "0.35rem" }}>
                    <span style={{ fontWeight: 600 }}>不足ひらがな:</span> {coverage.missing_hiragana}
                  </div>
                )}
                {coverage.missing_katakana && (
                  <div style={{ marginTop: "0.22rem" }}>
                    <span style={{ fontWeight: 600 }}>不足カタカナ:</span> {coverage.missing_katakana}
                  </div>
                )}
                {coverage.missing_kanji_core && (
                  <div style={{ marginTop: "0.22rem" }}>
                    <span style={{ fontWeight: 600 }}>不足漢字(コア):</span> {coverage.missing_kanji_core}
                  </div>
                )}
                {!!missingGuide && (
                  <button
                    className="tool-btn"
                    style={{ marginTop: "0.55rem", width: "100%" }}
                    onClick={copyMissingGuide}
                  >
                    不足文字をコピー
                  </button>
                )}
              </>
            ) : (
              <div style={{ color: "var(--text-secondary)" }}>
                カバレッジ情報を取得できませんでした。
              </div>
            )}
          </div>
        )}

        <div
          className={`paper style-${paperStyle} orientation-${orientation} ${selectedPaperDef.image ? "has-custom-image" : ""}`}
          style={{
            cursor: activeTool !== "pointer" ? "crosshair" : "default",
            ...customImageStyle,
          }}
          onPointerDown={handlePaperClick}
        >
          {items.map((rawItem) => {
            const item = punctuationRenderItem(rawItem);
            return (
            <div
              key={item.id}
              className={`draggable-wrapper ${selectedId === item.id ? "selected" : ""}`}
              style={{ top: item.y, left: item.x }}
              onPointerDown={() => setSelectedId(item.id)}
            >
              <div className="drag-header">
                <div
                  className="drag-handle"
                  onPointerDown={(e) => handlePointerDown(e, item.id)}
                  title="ドラッグして移動"
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="9" cy="12" r="1"></circle>
                    <circle cx="9" cy="5" r="1"></circle>
                    <circle cx="9" cy="19" r="1"></circle>
                    <circle cx="15" cy="12" r="1"></circle>
                    <circle cx="15" cy="5" r="1"></circle>
                    <circle cx="15" cy="19" r="1"></circle>
                  </svg>
                </div>
                <button
                  className="delete-btn"
                  onClick={() => deleteItem(item.id)}
                  title="削除"
                >
                  ×
                </button>
              </div>

              <div style={{ position: "relative" }}>
                {item.type === "text" ? (
                  isGeneratedPunctuationItem(item) ? (
                    <div
                      className="letter-input"
                      style={{
                        color: item.color,
                        fontSize: item.fontSize ? `${item.fontSize}px` : undefined,
                        lineHeight: 1,
                        letterSpacing: item.letterSpacing !== undefined ? `${item.letterSpacing}px` : undefined,
                        width: `${Math.max(18, item.boxWidth ?? item.fontSize ?? 18)}px`,
                        height: `${Math.max(18, item.boxHeight ?? item.fontSize ?? 18)}px`,
                        minWidth: 0,
                        minHeight: 0,
                        resize: "none",
                        overflow: "visible",
                        padding: 0,
                        borderColor: "transparent",
                        fontWeight: 800,
                        textAlign: "center",
                        textShadow: `0 0 0 ${item.color}, 0.35px 0 0 ${item.color}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        pointerEvents: "none",
                      }}
                    >
                      {SMALL_PUNCT_CHARS.has(item.text) || TALL_PUNCT_CHARS.has(item.text) ? (
                        <PunctuationMark ch={item.text} color={item.color} />
                      ) : (
                        item.text
                      )}
                    </div>
                  ) : (
                  <>
                    <textarea
                      className={`letter-input style-${paperStyle} ${item.isConverted && !isPunctuationOnlyText(item.text) ? "hidden-text" : ""}`}
                      style={{
                        color: item.color,
                        fontSize: item.fontSize
                          ? `${item.fontSize}px`
                          : undefined,
                        lineHeight:
                          item.lineHeight !== undefined
                            ? item.lineHeight
                            : item.fontSize && item.fontSize > 26
                              ? 1.5
                              : "2.5rem",
                        letterSpacing:
                          item.letterSpacing !== undefined
                            ? `${item.letterSpacing}px`
                            : undefined,
                      }}
                      value={item.text}
                      onChange={(e) => {
                        const val = e.target.value;
                        setItems((prev) =>
                          prev.map((it) =>
                            it.id === item.id
                              ? { ...it, text: val, isConverted: false, svg: "" }
                              : it,
                          ),
                        );
                      }}
                      onBlur={commitHistory}
                      placeholder="ここにメッセージを入力してください..."
                      disabled={item.isConverted || isProcessing}
                    />
                    {item.svg && (
                      <div
                        className={`handwriting-svg ${item.isConverted ? "visible" : ""}`}
                        style={{ color: item.color }}
                        dangerouslySetInnerHTML={{ __html: item.svg }}
                      />
                    )}
                  </>
                  )
                ) : item.type === "handwriting" ? (
                  <div
                    className={`handwriting-editor ${selectedId === item.id ? "selected" : ""} ${activeTool === "eraser" ? "eraser-mode" : ""}`}
                    style={{
                      width: `${handwritingBoxSize(item).width}px`,
                      height: `${handwritingBoxSize(item).height}px`,
                    }}
                  >
                    {(item.paths ?? []).length === 0 ? (
                      <div
                        className="handwriting-svg visible"
                        style={{ color: item.color, position: "absolute", inset: 0, padding: 0, width: "100%", height: "100%" }}
                        dangerouslySetInnerHTML={{ __html: item.svg }}
                      />
                    ) : (
                      <svg
                        width="100%"
                        height="100%"
                        viewBox={item.svgViewBox ?? "0 0 100 100"}
                        preserveAspectRatio="xMinYMin meet"
                      >
                        {(item.paths ?? []).map((p) => (
                          <path
                            key={p.id}
                            d={p.d}
                            fill="none"
                            stroke={item.color}
                            strokeWidth={p.strokeWidth}
                            vectorEffect="non-scaling-stroke"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            onPointerDown={(e) => {
                              if (activeTool !== "eraser") return;
                              e.preventDefault();
                              e.stopPropagation();
                              setItems((prev) =>
                                prev.map((it) =>
                                  it.id === item.id
                                    ? {
                                        ...it,
                                        paths: (it.paths ?? []).filter((sp) => sp.id !== p.id),
                                      }
                                    : it,
                                ),
                              );
                              commitHistory();
                            }}
                          />
                        ))}
                      </svg>
                    )}
                  </div>
                ) : (
                  <div className="shape-wrapper">
                    <ShapeRenderer type={item.type} color={item.color} />
                  </div>
                )}
              </div>
            </div>
          );})}
        </div>

        <div
          className="convert-btn-wrapper animate-fade-in"
          style={{
            animationDelay: "0.4s",
            animationFillMode: "both",
            position: "absolute",
            bottom: "2rem",
            right: "2rem",
            zIndex: 100,
          }}
        >
          <button
            className="convert-btn"
            onClick={handleConvert}
            disabled={!hasTextToConvert || isProcessing}
          >
            {isProcessing ? "変換中..." : "筆跡に変換する"}
          </button>
        </div>
      </div>
    </div>
  );
}
