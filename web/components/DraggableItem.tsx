import React from "react";
import { CanvasItem } from "../lib/types";
import { ShapeRenderer } from "./ShapeRenderer";
import { PunctuationMark } from "./PunctuationMark";
import {
  isGeneratedPunctuationItem,
  isPunctuationOnlyText,
  handwritingBoxSize,
  SMALL_PUNCT_CHARS,
  TALL_PUNCT_CHARS,
} from "../lib/canvasHelpers";

type Props = {
  item: CanvasItem;
  isSelected: boolean;
  selectedId: string | null;
  selectedIds: string[];
  setSelectedIds: React.Dispatch<React.SetStateAction<string[]>>;
  paperStyle: string;
  isProcessing: boolean;
  setItems: React.Dispatch<React.SetStateAction<CanvasItem[]>>;
  commitHistory: () => void;
  handlePointerDown: (e: React.PointerEvent, id: string) => void;
  handleResizeStart: (e: React.PointerEvent, id: string, dir: "nw" | "ne" | "se" | "sw") => void;
  handleRotateStart: (e: React.PointerEvent, id: string) => void;
  deleteItem: (id: string) => void;
  isPanningActive?: boolean;
};

export function DraggableItem({
  item,
  isSelected,
  selectedId,
  selectedIds,
  setSelectedIds,
  paperStyle,
  isProcessing,
  setItems,
  commitHistory,
  handlePointerDown,
  handleResizeStart,
  handleRotateStart,
  deleteItem,
  isPanningActive,
}: Props) {
  return (
    <div
      id={`wrapper-${item.id}`}
      className={`draggable-wrapper ${isSelected ? "selected" : ""}`}
      style={{
        top: item.y,
        left: item.x,
        transform: item.rotate ? `rotate(${item.rotate}deg)` : undefined,
        pointerEvents: isPanningActive ? "none" : "auto",
      }}
      onPointerDown={(e) => {
        e.stopPropagation();
        if (e.shiftKey) {
          setSelectedIds((prev) =>
            prev.includes(item.id)
              ? prev.filter((x) => x !== item.id)
              : [...prev, item.id]
          );
        } else {
          if (!selectedIds.includes(item.id)) {
            setSelectedIds([item.id]);
          }
        }
      }}
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
                className={`letter-input style-${paperStyle} ${
                  item.isConverted && !isPunctuationOnlyText(item.text) ? "hidden-text" : ""
                }`}
                style={{
                  width: item.boxWidth ? `${item.boxWidth}px` : undefined,
                  height: item.boxHeight ? `${item.boxHeight}px` : undefined,
                  color: item.color,
                  fontSize: item.fontSize ? `${item.fontSize}px` : undefined,
                  lineHeight:
                    item.lineHeight !== undefined
                      ? item.lineHeight
                      : item.fontSize && item.fontSize > 26
                      ? 1.5
                      : "2.5rem",
                  letterSpacing:
                    item.letterSpacing !== undefined ? `${item.letterSpacing}px` : undefined,
                }}
                value={item.text}
                onChange={(e) => {
                  const val = e.target.value;
                  setItems((prev) =>
                    prev.map((it) =>
                      it.id === item.id ? { ...it, text: val, isConverted: false, svg: "" } : it
                    )
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
            className={`handwriting-editor ${selectedId === item.id ? "selected" : ""}`}
            style={{
              width: `${handwritingBoxSize(item).width}px`,
              height: `${handwritingBoxSize(item).height}px`,
            }}
          >
            {(item.paths ?? []).length === 0 ? (
              <div
                className="handwriting-svg visible"
                style={{
                  color: item.color,
                  position: "absolute",
                  inset: 0,
                  padding: 0,
                  width: "100%",
                  height: "100%",
                }}
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
                    strokeOpacity={p.strokeOpacity}
                    vectorEffect="non-scaling-stroke"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                ))}
              </svg>
            )}
          </div>
        ) : (
          <div
            className="shape-wrapper"
            style={{
              width: item.boxWidth ? `${item.boxWidth}px` : undefined,
              height: item.boxHeight ? `${item.boxHeight}px` : undefined,
            }}
          >
            <ShapeRenderer type={item.type} color={item.color} />
          </div>
        )}
      </div>

      {/* Rotate Handle */}
      {selectedIds.length <= 1 && (
        <div
          className="rotate-handle"
          onPointerDown={(e) => handleRotateStart(e, item.id)}
          title="ドラッグして回転"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
        </div>
      )}

      {/* Corner Resize Handles / Selection Dots (PowerPoint style) */}
      {isSelected && (
        <>
          <div
            className="selection-corner-dot top-left"
            style={{ cursor: selectedId === item.id && selectedIds.length <= 1 ? "nwse-resize" : "default" }}
            onPointerDown={
              selectedId === item.id && selectedIds.length <= 1
                ? (e) => handleResizeStart(e, item.id, "nw")
                : undefined
            }
          />
          <div
            className="selection-corner-dot top-right"
            style={{ cursor: selectedId === item.id && selectedIds.length <= 1 ? "nesw-resize" : "default" }}
            onPointerDown={
              selectedId === item.id && selectedIds.length <= 1
                ? (e) => handleResizeStart(e, item.id, "ne")
                : undefined
            }
          />
          <div
            className="selection-corner-dot bottom-left"
            style={{ cursor: selectedId === item.id && selectedIds.length <= 1 ? "nesw-resize" : "default" }}
            onPointerDown={
              selectedId === item.id && selectedIds.length <= 1
                ? (e) => handleResizeStart(e, item.id, "sw")
                : undefined
            }
          />
          <div
            className="selection-corner-dot bottom-right"
            style={{ cursor: selectedId === item.id && selectedIds.length <= 1 ? "nwse-resize" : "default" }}
            onPointerDown={
              selectedId === item.id && selectedIds.length <= 1
                ? (e) => handleResizeStart(e, item.id, "se")
                : undefined
            }
          />
        </>
      )}
    </div>
  );
}
