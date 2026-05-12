import { useEffect, useState } from "react";
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

  const [textColor, setTextColor] = useState(COLORS[0].color);
  const [textSize, setTextSize] = useState(24);
  const [paperStyle, setPaperStyle] = useState(PAPERS[0].id);

  const [items, setItems] = useState<CanvasItem[]>([
    { id: '1', type: 'text', x: 40, y: 40, text: '', svg: '', color: COLORS[0].color, isConverted: false, fontSize: 24 }
  ]);
  const [selectedId, setSelectedId] = useState<string | null>('1');

  // Drag logic states
  const [isDragging, setIsDragging] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [activeTool, setActiveTool] = useState<CanvasItem["type"] | "pointer">("pointer");

  useEffect(() => {
    const handleGlobalMove = (e: PointerEvent) => {
      if (isDragging) {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        setItems((prev) => prev.map((it) => it.id === isDragging ? { ...it, x: it.x + dx, y: it.y + dy } : it));
        setDragStart({ x: e.clientX, y: e.clientY });
      }
    };
    const handleGlobalUp = () => {
      setIsDragging(null);
    };

    if (isDragging) {
      window.addEventListener('pointermove', handleGlobalMove);
      window.addEventListener('pointerup', handleGlobalUp);
    }
    return () => {
      window.removeEventListener('pointermove', handleGlobalMove);
      window.removeEventListener('pointerup', handleGlobalUp);
    }
  }, [isDragging, dragStart]);

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
        };
        setItems([...items, newItem]);
        setSelectedId(newItem.id);
        
        setActiveTool("pointer");
      } else {
        setSelectedId(null);
      }
    }
  };

  const deleteItem = (id: string) => {
    setItems(items.filter(it => it.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  const selectedItem = items.find((it) => it.id === selectedId);
  const currentColor = selectedItem ? selectedItem.color : textColor;
  const currentFontSize = selectedItem && selectedItem.type === "text" && selectedItem.fontSize ? selectedItem.fontSize : textSize;

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
          it.id === selectedId && it.type === "text" ? { ...it, fontSize: newSize } : it,
        ),
      );
    }
  };

  async function handleConvert() {
    const textsToConvert = items.filter(it => it.type === 'text' && !it.isConverted && it.text.trim() !== '');
    if (textsToConvert.length === 0) return;

    setIsProcessing(true);
    try {
      if (token && userId && styleId) {
        let newItems = [...items];
        for (let i = 0; i < newItems.length; i++) {
          const item = newItems[i];
          if (item.type === 'text' && !item.isConverted && item.text.trim() !== '') {
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
              newItems[i] = { ...item, svg: body.svg, isConverted: true };
            }
          }
        }
        setItems(newItems);
      } else {
        setItems(items.map(it => it.type === 'text' && it.text.trim() !== '' ? { ...it, isConverted: true } : it));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  }

  const hasUnconvertedText = items.some(it => it.type === 'text' && !it.isConverted && it.text.trim() !== '');

  return (
    <div className="canvas-screen animate-fade-in">
      <div className="sidebar animate-slide-up">
        <div className="sidebar-section">
          <h3>便箋デザイン</h3>
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

        <hr className="sidebar-divider" />

        <div className="sidebar-section">
          <h3>アイテム配置</h3>
          <div className="tool-grid">
            <button
              className={`tool-btn ${activeTool === "pointer" ? "active" : ""}`}
              onClick={() => setActiveTool("pointer")}
              title="選択・移動"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path><path d="M13 13l6 6"></path></svg>
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
          <h3>文字の大きさ</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>小</span>
            <input
              type="range"
              min="16"
              max="80"
              value={currentFontSize}
              onChange={(e) => handleFontSizeChange(parseInt(e.target.value))}
              className="sidebar-slider"
            />
            <span style={{ fontSize: '1.2rem', color: 'var(--text-secondary)' }}>大</span>
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
                onClick={() => handleColorChange(c.color)}
              />
            ))}
            <div 
              className="color-picker-wrapper"
              title="自由な色を選ぶ"
            >
              <input
                type="color"
                className="color-picker-input"
                value={currentColor}
                onChange={(e) => handleColorChange(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="canvas-main">
        <div
          className={`paper style-${paperStyle} animate-slide-up`}
          style={{ cursor: activeTool !== "pointer" ? "crosshair" : "default" }}
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
                        fontSize: item.fontSize ? `${item.fontSize}px` : undefined,
                        lineHeight: item.fontSize && item.fontSize > 26 ? 1.5 : "2.5rem",
                      }}
                      value={item.text}
                      onChange={(e) => {
                        const val = e.target.value;
                        setItems(
                          items.map((it) =>
                            it.id === item.id ? { ...it, text: val } : it,
                          ),
                        );
                      }}
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
          style={{ animationDelay: "0.4s", animationFillMode: "both" }}
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
