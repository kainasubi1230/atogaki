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
      (it) => it.type === "text" && it.text.trim() !== "" && !it.isConverted,
    );
    if (textsToConvert.length === 0) return;

    setIsProcessing(true);
    try {
      if (!token || !userId || !styleId) {
        alert("スタイル準備ができていません。先に画像アップロードをやり直してください。");
        return;
      }

      let newItems: CanvasItem[] = [];
      let convertedCount = 0;
      let failedCount = 0;
      let lastErrorDetail = "";
      for (const item of items) {
        if (item.type !== "text" || item.text.trim() === "" || item.isConverted) {
          newItems.push(item);
          continue;
        }

        const lines = item.text.split("\n");
        const generatedChars: CanvasItem[] = [];
        const baseFont = item.fontSize ?? 24;
        const lineHeightPx = (item.lineHeight ?? 1.5) * baseFont;
        const letterSpace = item.letterSpacing ?? 0;
        let currentY = item.y;
        for (const line of lines) {
          let currentX = item.x;
          for (const ch of line) {
            if (ch.trim() === "") {
              currentX += baseFont * 0.6 + letterSpace;
              continue;
            }
            const result = await postJSON(
              "/generate",
              {
                user_id: userId,
                style_id: styleId,
                text: ch,
                purpose: "accessibility",
              },
              token,
            );
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
            const targetW = Math.max(18, Math.min(220, targetH * ratio));
            generatedChars.push({
              id: `${item.id}_hw_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
              type: "handwriting",
              x: currentX,
              y: currentY,
              text: ch,
              svg: body.svg,
              color: item.color,
              isConverted: true,
              paths: parsed?.paths ?? [],
              svgViewBox: parsed?.viewBox ?? fallbackDims?.viewBox ?? "0 0 100 100",
              boxWidth: targetW,
              boxHeight: targetH,
            });
            convertedCount += 1;
            currentX += targetW + Math.max(2, letterSpace);
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
    (it) => it.type === "text" && it.text.trim() !== "" && !it.isConverted,
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
          {items.map((item) => (
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
                  <>
                    <textarea
                      className={`letter-input style-${paperStyle} ${item.isConverted ? "hidden-text" : ""}`}
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
                ) : item.type === "handwriting" ? (
                  <div
                    className={`handwriting-editor ${selectedId === item.id ? "selected" : ""} ${activeTool === "eraser" ? "eraser-mode" : ""}`}
                    style={{
                      width: `${item.boxWidth ?? 80}px`,
                      height: `${item.boxHeight ?? 56}px`,
                    }}
                  >
                    {(item.paths ?? []).length === 0 ? (
                      <div
                        className="handwriting-svg visible"
                        style={{ color: item.color, position: "relative", inset: 0, padding: 0 }}
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
          ))}
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
