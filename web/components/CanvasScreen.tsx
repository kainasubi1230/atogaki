import { useEffect, useState, useCallback } from "react";
import { CanvasItem } from "../lib/types";
import { COLORS, PAPERS } from "../lib/constants";
import { postJSON } from "../lib/api";
import { ShapeRenderer } from "./ShapeRenderer";

type Props = {
  token: string;
  userId: number | null;
  styleId: number | null;
};

export function CanvasScreen({ token, userId, styleId }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(true);

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
  const [activeTool, setActiveTool] = useState<CanvasItem["type"] | "pointer">(
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
      if (activeTool !== "pointer") {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const newItem: CanvasItem = {
          id: Date.now().toString(),
          type: activeTool as CanvasItem["type"],
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
      (it) => it.type === "text" && !it.isConverted && it.text.trim() !== "",
    );
    if (textsToConvert.length === 0) return;

    setIsProcessing(true);
    try {
      if (!token || !userId || !styleId) {
        alert("スタイル準備ができていません。先に画像アップロードをやり直してください。");
        return;
      }

      let newItems = [...items];
      let convertedCount = 0;
      let failedCount = 0;
      let lastErrorDetail = "";
      for (let i = 0; i < newItems.length; i++) {
        const item = newItems[i];
        if (
          item.type === "text" &&
          !item.isConverted &&
          item.text.trim() !== ""
        ) {
          const result = await postJSON(
            "/generate",
            {
              user_id: userId,
              style_id: styleId,
              text: item.text,
              purpose: "accessibility",
            },
            token,
          );
          if (result.status === 200) {
            const body = result.body as { svg: string };
            if (body.svg && body.svg.trim() !== "") {
              newItems[i] = { ...item, svg: body.svg, isConverted: true };
              convertedCount += 1;
            } else {
              failedCount += 1;
              lastErrorDetail = "empty_svg";
            }
          } else {
            failedCount += 1;
            const body = result.body as { detail?: string };
            lastErrorDetail = body?.detail ?? `http_${result.status}`;
          }
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

  const hasUnconvertedText = items.some(
    (it) => it.type === "text" && !it.isConverted && it.text.trim() !== "",
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
                            it.id === item.id ? { ...it, text: val } : it,
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
            disabled={!hasUnconvertedText || isProcessing}
          >
            {isProcessing ? "変換中..." : "筆跡に変換する"}
          </button>
        </div>
      </div>
    </div>
  );
}
