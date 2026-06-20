import { useEffect, useState, useCallback, useRef } from "react";
import { CanvasItem } from "../lib/types";
import { COLORS, PAPERS } from "../lib/constants";
import { getJSON, postJSON } from "../lib/api";
import { useCanvasHistory } from "../hooks/useCanvasHistory";
import { useCanvasInteraction } from "../hooks/useCanvasInteraction";
import { ShojiOverlay } from "./ShojiOverlay";
import { CanvasSidebar } from "./CanvasSidebar";
import { DraggableItem } from "./DraggableItem";
import { HIDDEN_MOTIONS } from "../lib/hiddenMotions";
import {
  isAsciiRenderableText,
  isLatinChar,
  isPunctuation,
  isSmallKana,
  isFullSizeKana,
  isSmallKatakana,
  smallKanaScale,
  isGeneratedPunctuationItem,
  isPunctuationOnlyText,
  getLatinCharLayout,
  parseSvgDimensions,
  parseHandwritingSvg,
  punctuationRenderItem,
  SMALL_YOON_CHARS,
  SMALL_SOKUON_CHARS,
  SMALL_PUNCT_CHARS,
  MEDIUM_PUNCT_CHARS,
  TALL_PUNCT_CHARS,
  BRACKET_PUNCT_CHARS,
  DASH_PUNCT_CHARS,
  OPERATOR_PUNCT_CHARS,
  QUOTE_PUNCT_CHARS,
} from "../lib/canvasHelpers";

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

export function CanvasScreen({ token, userId, styleId }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<"paper" | "write">("paper");
  const [customColor, setCustomColor] = useState("#ff6600");
  const [savedColors, setSavedColors] = useState<string[]>([]);
  const [coverage, setCoverage] = useState<StyleCoverage | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(false);
  const [activeMotions, setActiveMotions] = useState<string[]>([]);

  const [isExporting, setIsExporting] = useState(false);
  const [textColor, setTextColor] = useState(COLORS[0].color);
  const [textSize, setTextSize] = useState(24);
  const [paperStyle, setPaperStyle] = useState(PAPERS[0].id);
  const [orientation, setOrientation] = useState<"portrait" | "landscape">("portrait");

  const [textLetterSpacing, setTextLetterSpacing] = useState(0);
  const [textLineHeight, setTextLineHeight] = useState(1.5);

  const { items, setItems, commitHistory } = useCanvasHistory();

  const {
    selectedIds,
    setSelectedIds,
    selectedId,
    setSelectedId,
    marqueeStart,
    marqueeEnd,
    activeTool,
    setActiveTool,
    handlePointerDown,
    handleResizeStart,
    handleRotateStart,
    handlePaperClick,
    handlePaperPointerMove,
    handlePaperPointerUp,
    deleteItem,
    zoom,
    setZoom,
    panOffset,
    setPanOffset,
    isSpacePanning,
    isPanning,
  } = useCanvasInteraction({
    items,
    setItems,
    commitHistory,
    textColor,
    textSize,
    textLetterSpacing,
    textLineHeight,
  });

  const getCursorStyle = () => {
    if (activeTool === "hand" || isSpacePanning) {
      return isPanning ? "grabbing" : "grab";
    }
    return activeTool !== "pointer" ? "crosshair" : "default";
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(3, Math.round((prev + 0.1) * 10) / 10));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(0.3, Math.round((prev - 0.1) * 10) / 10));
  };

  const handleZoomReset = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const [isEditingZoom, setIsEditingZoom] = useState(false);
  const [zoomInputValue, setZoomInputValue] = useState("");

  const handleZoomIndicatorClick = () => {
    setZoomInputValue(String(Math.round(zoom * 100)));
    setIsEditingZoom(true);
  };

  const handleZoomInputSubmit = () => {
    setIsEditingZoom(false);
    const parsed = parseInt(zoomInputValue.replace(/%/g, "").trim(), 10);
    if (!isNaN(parsed)) {
      const clamped = Math.min(300, Math.max(30, parsed));
      setZoom(clamped / 100);
    }
  };

  const handleZoomInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleZoomInputSubmit();
    } else if (e.key === "Escape") {
      setIsEditingZoom(false);
    }
  };

  // Listen to wheel event for Ctrl/Cmd + wheel zoom
  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.08 : 0.92;
        setZoom((prev) => {
          const next = prev * factor;
          return Math.min(3, Math.max(0.3, Math.round(next * 100) / 100));
        });
      }
    };

    const mainEl = document.querySelector(".canvas-main");
    if (mainEl) {
      mainEl.addEventListener("wheel", handleWheel as any, { passive: false });
    }
    return () => {
      if (mainEl) {
        mainEl.removeEventListener("wheel", handleWheel as any);
      }
    };
  }, [setZoom]);

  // Sync color states with local storage
  useEffect(() => {
    const savedColor = localStorage.getItem("machine_selected_color_v1");
    if (savedColor) {
      setTextColor(savedColor);
    }
    const savedCustomColor = localStorage.getItem("machine_custom_color_v1");
    if (savedCustomColor) {
      setCustomColor(savedCustomColor);
    }
    const savedColorsRaw = localStorage.getItem("machine_saved_colors_v1");
    if (savedColorsRaw) {
      try {
        const parsed = JSON.parse(savedColorsRaw);
        if (Array.isArray(parsed)) {
          setSavedColors(parsed.filter((c) => typeof c === "string"));
        }
      } catch (e) {
        console.error("Failed to parse saved colors", e);
      }
    }
  }, []);

  const saveColor = (color: string) => {
    if (savedColors.includes(color)) return;
    const next = [...savedColors, color];
    setSavedColors(next);
    localStorage.setItem("machine_saved_colors_v1", JSON.stringify(next));
  };

  const deleteSavedColor = (color: string) => {
    const next = savedColors.filter((c) => c !== color);
    setSavedColors(next);
    localStorage.setItem("machine_saved_colors_v1", JSON.stringify(next));
  };

  const paperRef = useRef<HTMLDivElement>(null);

  const handleDownload = async (format: "png" | "svg") => {
    if (!paperRef.current) return;
    setIsExporting(true);

    await new Promise((resolve) => setTimeout(resolve, 80));

    const stylesheets = Array.from(document.querySelectorAll("link[rel='stylesheet'], style"));
    const detachedStylesheets: { element: Element; parent: Node; nextSibling: Node | null }[] = [];

    stylesheets.forEach((el) => {
      try {
        const sheet = (el as HTMLStyleElement | HTMLLinkElement).sheet;
        if (sheet) {
          const _ = sheet.cssRules;
        }
      } catch (error) {
        const parent = el.parentNode;
        if (parent) {
          detachedStylesheets.push({
            element: el,
            parent,
            nextSibling: el.nextSibling,
          });
          parent.removeChild(el);
        }
      }
    });

    try {
      const { toPng, toSvg } = await import("html-to-image");
      if (format === "png") {
        const dataUrl = await toPng(paperRef.current, {
          quality: 0.98,
          backgroundColor: "#faf8f5",
        });
        const link = document.createElement("a");
        link.download = `handwriting-${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
      } else {
        const dataUrl = await toSvg(paperRef.current, {
          backgroundColor: "#faf8f5",
        });
        const link = document.createElement("a");
        link.download = `handwriting-${Date.now()}.svg`;
        link.href = dataUrl;
        link.click();
      }
    } catch (error) {
      console.error("Export failed:", error);
      alert("画像のダウンロードに失敗しました。");
    } finally {
      detachedStylesheets.forEach(({ element, parent, nextSibling }) => {
        parent.insertBefore(element, nextSibling);
      });
      setIsExporting(false);
    }
  };

  const selectedItem = items.find((it) => it.id === selectedId);
  const currentColor = selectedItem ? selectedItem.color : textColor;
  const isCustomColorActive = !COLORS.some((c) => c.color === currentColor);

  let customColorIconColor = "#5c665c";
  if (isCustomColorActive) {
    const hex = currentColor.replace("#", "");
    if (hex.length === 6) {
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      const yiq = (r * 299 + g * 587 + b * 114) / 1000;
      customColorIconColor = yiq >= 128 ? "#000" : "#fff";
    } else {
      customColorIconColor = "#fff";
    }
  }

  const currentFontSize =
    selectedItem && selectedItem.type === "text" && selectedItem.fontSize !== undefined
      ? selectedItem.fontSize
      : textSize;
  const currentLetterSpacing =
    selectedItem && selectedItem.type === "text" && selectedItem.letterSpacing !== undefined
      ? selectedItem.letterSpacing
      : textLetterSpacing;
  const currentLineHeight =
    selectedItem && selectedItem.type === "text" && selectedItem.lineHeight !== undefined
      ? selectedItem.lineHeight
      : textLineHeight;

  const handleColorChange = (newColor: string) => {
    setTextColor(newColor);
    localStorage.setItem("machine_selected_color_v1", newColor);
    if (selectedIds.length > 0) {
      setItems((prev) =>
        prev.map((it) => (selectedIds.includes(it.id) ? { ...it, color: newColor } : it))
      );
    }
  };

  const handleFontSizeChange = (newSize: number) => {
    setTextSize(newSize);
    if (selectedIds.length > 0) {
      setItems((prev) =>
        prev.map((it) =>
          selectedIds.includes(it.id) && it.type === "text" ? { ...it, fontSize: newSize } : it
        )
      );
    }
  };

  const handleLetterSpacingChange = (newVal: number) => {
    setTextLetterSpacing(newVal);
    if (selectedIds.length > 0) {
      setItems((prev) =>
        prev.map((it) =>
          selectedIds.includes(it.id) && it.type === "text" ? { ...it, letterSpacing: newVal } : it
        )
      );
    }
  };

  const handleLineHeightChange = (newVal: number) => {
    setTextLineHeight(newVal);
    if (selectedIds.length > 0) {
      setItems((prev) =>
        prev.map((it) =>
          selectedIds.includes(it.id) && it.type === "text" ? { ...it, lineHeight: newVal } : it
        )
      );
    }
  };

  async function handleConvert() {
    const textsToConvert = items.filter(
      (it) => it.type === "text" && it.text.trim() !== "" && !it.isConverted
    );
    if (textsToConvert.length === 0) return;

    // Detect trigger words for hidden motions
    const combinedText = textsToConvert.map((it) => it.text).join(" ");
    const triggered: string[] = [];
    HIDDEN_MOTIONS.forEach((motion) => {
      const matches = motion.keywords.some((keyword) => combinedText.includes(keyword));
      if (matches) {
        triggered.push(motion.id);
      }
    });
    setActiveMotions(triggered);

    setIsProcessing(true);
    const startTime = Date.now();
    try {
      const canConvertWithoutStyle = textsToConvert.every((item) => isAsciiRenderableText(item.text));
      if (!token || !userId) {
        alert("スタイル準備ができていません。先に画像アップロードをやり直してください。");
        return;
      }

      let useFallbackDefaultStyle = false;
      if (!styleId && !canConvertWithoutStyle) {
        useFallbackDefaultStyle = true;
      }

      let newItems: CanvasItem[] = [];
      let convertedCount = 0;
      let failedCount = 0;
      let lastErrorDetail = "";
      const useRuntimePunctuation = true;
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
                  style_id: useFallbackDefaultStyle ? 0 : styleId ?? 0,
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
              const targetH = Math.max(22, baseFont * 1.75);
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
            const isJapanese = !isLatinChar(ch) && !isPunctuation(ch);
            const parsed = parseHandwritingSvg(body.svg, isJapanese);
            const fallbackDims = parseSvgDimensions(body.svg);
            const targetH = Math.max(22, baseFont * 1.75);
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
              const scale = smallKanaScale(ch);
              targetW *= scale;
              const smallH = targetH * scale;
              advanceW = targetW;
              let shiftX = targetW * 0.22;
              let shiftY = smallH * 0.28;
              const smallKatakana = isSmallKatakana(ch);

              if (previousPlacedChar && isFullSizeKana(previousPlacedChar)) {
                const prevFullKatakana = isFullSizeKana(previousPlacedChar) && !isSmallKana(previousPlacedChar);
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
            } else if (isLatinChar(ch)) {
              const layout = getLatinCharLayout(ch, targetH, targetW);
              targetW = layout.boxW;
              boxH = layout.boxH;
              drawX += layout.shiftX;
              drawY += layout.shiftY;
              advanceW = layout.advanceW;
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
      const elapsed = Date.now() - startTime;
      const minTime = 1200;
      if (elapsed < minTime) {
        await new Promise((resolve) => setTimeout(resolve, minTime - elapsed));
      }
      setIsProcessing(false);
      // Wait for doors to open completely before resetting motions so they don't pop out of view instantly
      setTimeout(() => {
        setActiveMotions([]);
      }, 800);
    }
  }

  const hasTextToConvert = items.some(
    (it) => it.type === "text" && it.text.trim() !== "" && !it.isConverted
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
      <CanvasSidebar
        isMenuOpen={isMenuOpen}
        setIsMenuOpen={setIsMenuOpen}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        paperStyle={paperStyle}
        setPaperStyle={setPaperStyle}
        orientation={orientation}
        setOrientation={setOrientation}
        activeTool={activeTool}
        setActiveTool={setActiveTool}
        currentFontSize={currentFontSize}
        handleFontSizeChange={handleFontSizeChange}
        currentLetterSpacing={currentLetterSpacing}
        handleLetterSpacingChange={handleLetterSpacingChange}
        currentLineHeight={currentLineHeight}
        handleLineHeightChange={handleLineHeightChange}
        commitHistory={commitHistory}
        currentColor={currentColor}
        handleColorChange={handleColorChange}
        customColor={customColor}
        setCustomColor={setCustomColor}
        isCustomColorActive={isCustomColorActive}
        customColorIconColor={customColorIconColor}
        savedColors={savedColors}
        saveColor={saveColor}
        deleteSavedColor={deleteSavedColor}
        handleDownload={handleDownload}
        isProcessing={isProcessing}
      />

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
          ref={paperRef}
          className={`paper style-${paperStyle} orientation-${orientation} ${
            selectedPaperDef.image ? "has-custom-image" : ""
          } ${isExporting ? "exporting" : ""}`}
          style={{
            cursor: getCursorStyle(),
            transform: isExporting
              ? "none"
              : `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            transition: isPanning ? "none" : "transform 0.15s ease-out",
            ...customImageStyle,
          }}
          onPointerDown={handlePaperClick}
          onPointerMove={handlePaperPointerMove}
          onPointerUp={handlePaperPointerUp}
        >
          {items.map((rawItem) => {
            const item = punctuationRenderItem(rawItem);
            const isSelected = selectedIds.includes(item.id);
            return (
              <DraggableItem
                key={item.id}
                item={item}
                isSelected={isSelected}
                selectedId={selectedId}
                selectedIds={selectedIds}
                setSelectedIds={setSelectedIds}
                paperStyle={paperStyle}
                isProcessing={isProcessing}
                setItems={setItems}
                commitHistory={commitHistory}
                handlePointerDown={handlePointerDown}
                handleResizeStart={handleResizeStart}
                handleRotateStart={handleRotateStart}
                deleteItem={deleteItem}
                isPanningActive={activeTool === "hand" || isSpacePanning}
              />
            );
          })}

          {marqueeStart && marqueeEnd && (
            <div
              className="selection-marquee"
              style={{
                left: `${Math.min(marqueeStart.x, marqueeEnd.x)}px`,
                top: `${Math.min(marqueeStart.y, marqueeEnd.y)}px`,
                width: `${Math.abs(marqueeStart.x - marqueeEnd.x)}px`,
                height: `${Math.abs(marqueeStart.y - marqueeEnd.y)}px`,
              }}
            />
          )}
        </div>

        {/* Floating Zoom and Panning controls */}
        <div
          className="zoom-panel animate-fade-in"
          style={{
            animationDelay: "0.4s",
            animationFillMode: "both",
          }}
        >
          <button
            className="zoom-panel-btn"
            onClick={handleZoomOut}
            title="縮小 (Ctrl + スクロール)"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
          </button>
          
          {isEditingZoom ? (
            <input
              type="text"
              className="zoom-panel-input"
              value={zoomInputValue}
              onChange={(e) => setZoomInputValue(e.target.value)}
              onBlur={handleZoomInputSubmit}
              onKeyDown={handleZoomInputKeyDown}
              autoFocus
              style={{
                width: "48px",
                textAlign: "center",
                fontSize: "0.85rem",
                fontFamily: "inherit",
                fontWeight: 600,
                border: "1px solid rgba(0, 0, 0, 0.15)",
                borderRadius: "4px",
                background: "rgba(0, 0, 0, 0.03)",
                padding: "2px 0",
                outline: "none",
                color: "var(--text-primary)",
              }}
            />
          ) : (
            <div
              className="zoom-panel-indicator"
              onClick={handleZoomIndicatorClick}
              onDoubleClick={handleZoomReset}
              title="クリックして数値を入力 / ダブルクリックでリセット"
            >
              {Math.round(zoom * 100)}%
            </div>
          )}

          <button
            className="zoom-panel-btn"
            onClick={handleZoomIn}
            title="拡大 (Ctrl + スクロール)"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
          </button>

          <div className="zoom-panel-divider" />

          <button
            className={`zoom-panel-btn ${activeTool === "hand" ? "active" : ""}`}
            onClick={() => setActiveTool(activeTool === "hand" ? "pointer" : "hand")}
            title="手のひらツール (Spaceキーを押しながらドラッグでも移動可能)"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v5"></path>
              <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v6"></path>
              <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v4.5"></path>
              <path d="M6 10a2 2 0 0 0-2 2v5a7 7 0 0 0 7 7h1a7 7 0 0 0 7-7v-6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2"></path>
            </svg>
          </button>

          <button
            className="zoom-panel-btn"
            onClick={handleZoomReset}
            title="ズームと位置をリセット"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 3h6v6M9 21H3v-6M21 9V3h-6M3 15v6h6M21 3l-7 7M3 21l7-7" />
            </svg>
          </button>
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

      <ShojiOverlay isProcessing={isProcessing} activeMotions={activeMotions} />
    </div>
  );
}
