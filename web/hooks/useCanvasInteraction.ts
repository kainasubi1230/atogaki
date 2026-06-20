import React, { useState, useEffect, useCallback } from "react";
import { CanvasItem } from "../lib/types";
import { getItemSize } from "../lib/canvasHelpers";

type InteractionProps = {
  items: CanvasItem[];
  setItems: React.Dispatch<React.SetStateAction<CanvasItem[]>>;
  commitHistory: () => void;
  textColor: string;
  textSize: number;
  textLetterSpacing: number;
  textLineHeight: number;
};

export function useCanvasInteraction({
  items,
  setItems,
  commitHistory,
  textColor,
  textSize,
  textLetterSpacing,
  textLineHeight,
}: InteractionProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>(["1"]);
  const [marqueeStart, setMarqueeStart] = useState<{ x: number; y: number } | null>(null);
  const [marqueeEnd, setMarqueeEnd] = useState<{ x: number; y: number } | null>(null);

  const [isDragging, setIsDragging] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isRotating, setIsRotating] = useState<string | null>(null);
  const [rotateStart, setRotateStart] = useState({ pointerAngle: 0, startRotation: 0, centerX: 0, centerY: 0 });

  const [isResizing, setIsResizing] = useState<string | null>(null);
  const [resizeDirection, setResizeDirection] = useState<'nw' | 'ne' | 'se' | 'sw' | null>(null);
  const [resizeStart, setResizeStart] = useState({
    x: 0,
    y: 0,
    startX: 0,
    startY: 0,
    startWidth: 0,
    startHeight: 0,
    rotate: 0,
  });

  const [activeTool, setActiveTool] = useState<
    "pointer" | "text" | "square" | "circle" | "triangle" | "arrow" | "hand"
  >("pointer");

  // Zoom and panning states
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isSpacePanning, setIsSpacePanning] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [panStartOffset, setPanStartOffset] = useState({ x: 0, y: 0 });

  const selectedId = selectedIds[selectedIds.length - 1] || null;
  
  const setSelectedId = useCallback((id: string | null) => {
    setSelectedIds(id ? [id] : []);
  }, []);

  // Keyboard shortcut for deleting selected items and Space key for panning
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Backspace" || e.key === "Delete") {
        const activeEl = document.activeElement;
        if (
          activeEl &&
          (activeEl.tagName === "INPUT" ||
            activeEl.tagName === "TEXTAREA" ||
            activeEl.getAttribute("contenteditable") === "true")
        ) {
          return;
        }
        if (selectedIds.length > 0) {
          e.preventDefault();
          setItems((prev) => prev.filter((it) => !selectedIds.includes(it.id)));
          setSelectedIds([]);
          commitHistory();
        }
      }

      if (e.code === "Space") {
        const activeEl = document.activeElement;
        if (
          activeEl &&
          (activeEl.tagName === "INPUT" ||
            activeEl.tagName === "TEXTAREA" ||
            activeEl.getAttribute("contenteditable") === "true")
        ) {
          return;
        }
        e.preventDefault();
        setIsSpacePanning(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setIsSpacePanning(false);
      }
    };

    const handleBlur = () => {
      setIsSpacePanning(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, [selectedIds, commitHistory, setItems]);

  // Global pointer listeners for active operations
  useEffect(() => {
    const handleGlobalMove = (e: PointerEvent) => {
      if (isDragging) {
        const dx = (e.clientX - dragStart.x) / zoom;
        const dy = (e.clientY - dragStart.y) / zoom;
        setItems((prev) =>
          prev.map((it) =>
            selectedIds.includes(it.id)
              ? { ...it, x: it.x + dx, y: it.y + dy }
              : it.id === isDragging
              ? { ...it, x: it.x + dx, y: it.y + dy }
              : it,
          ),
        );
        setDragStart({ x: e.clientX, y: e.clientY });
      } else if (isRotating) {
        const dx = e.clientX - rotateStart.centerX;
        const dy = e.clientY - rotateStart.centerY;
        const currentPointerAngle = (Math.atan2(dy, dx) * 180) / Math.PI;
        const diff = currentPointerAngle - rotateStart.pointerAngle;
        let nextRotation = Math.round(rotateStart.startRotation + diff);

        if (nextRotation > 180) nextRotation -= 360;
        if (nextRotation < -180) nextRotation += 360;

        setItems((prev) =>
          prev.map((it) =>
            it.id === isRotating ? { ...it, rotate: nextRotation } : it,
          ),
        );
      } else if (isResizing && resizeDirection) {
        const dx = (e.clientX - resizeStart.x) / zoom;
        const dy = (e.clientY - resizeStart.y) / zoom;

        const rad = (resizeStart.rotate * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);

        // Project global move vector into local space of the rotated object
        const dxLocal = dx * cos + dy * sin;
        const dyLocal = -dx * sin + dy * cos;

        let newWidth = resizeStart.startWidth;
        let newHeight = resizeStart.startHeight;

        let deltaWidth = 0;
        let deltaHeight = 0;

        if (resizeDirection === "se") {
          deltaWidth = dxLocal;
          deltaHeight = dyLocal;
        } else if (resizeDirection === "sw") {
          deltaWidth = -dxLocal;
          deltaHeight = dyLocal;
        } else if (resizeDirection === "ne") {
          deltaWidth = dxLocal;
          deltaHeight = -dyLocal;
        } else if (resizeDirection === "nw") {
          deltaWidth = -dxLocal;
          deltaHeight = -dyLocal;
        }

        if (e.shiftKey) {
          const scaleX = (resizeStart.startWidth + deltaWidth) / resizeStart.startWidth;
          const scaleY = (resizeStart.startHeight + deltaHeight) / resizeStart.startHeight;

          const diffX = Math.abs(scaleX - 1);
          const diffY = Math.abs(scaleY - 1);
          let scale = diffX > diffY ? scaleX : scaleY;

          const limitScaleX = 30 / resizeStart.startWidth;
          const limitScaleY = 20 / resizeStart.startHeight;
          const limitScale = Math.max(limitScaleX, limitScaleY);
          if (scale < limitScale) {
            scale = limitScale;
          }

          newWidth = resizeStart.startWidth * scale;
          newHeight = resizeStart.startHeight * scale;
        } else {
          newWidth = Math.max(30, resizeStart.startWidth + deltaWidth);
          newHeight = Math.max(20, resizeStart.startHeight + deltaHeight);
        }

        let anchorLocal = { x: 0, y: 0 };
        let anchorLocalNew = { x: 0, y: 0 };

        if (resizeDirection === "se") {
          anchorLocal = { x: -resizeStart.startWidth / 2, y: -resizeStart.startHeight / 2 };
          anchorLocalNew = { x: -newWidth / 2, y: -newHeight / 2 };
        } else if (resizeDirection === "sw") {
          anchorLocal = { x: resizeStart.startWidth / 2, y: -resizeStart.startHeight / 2 };
          anchorLocalNew = { x: newWidth / 2, y: -newHeight / 2 };
        } else if (resizeDirection === "ne") {
          anchorLocal = { x: -resizeStart.startWidth / 2, y: resizeStart.startHeight / 2 };
          anchorLocalNew = { x: -newWidth / 2, y: newHeight / 2 };
        } else if (resizeDirection === "nw") {
          anchorLocal = { x: resizeStart.startWidth / 2, y: resizeStart.startHeight / 2 };
          anchorLocalNew = { x: newWidth / 2, y: newHeight / 2 };
        }

        // Global anchor point (which shouldn't move)
        const startCenter = {
          x: resizeStart.startX + resizeStart.startWidth / 2,
          y: resizeStart.startY + resizeStart.startHeight / 2,
        };
        const anchorGlobal = {
          x: startCenter.x + (anchorLocal.x * cos - anchorLocal.y * sin),
          y: startCenter.y + (anchorLocal.x * sin + anchorLocal.y * cos),
        };

        // Solve for new center such that the global anchor point is preserved
        const newCenterX = anchorGlobal.x - (anchorLocalNew.x * cos - anchorLocalNew.y * sin);
        const newCenterY = anchorGlobal.y - (anchorLocalNew.x * sin + anchorLocalNew.y * cos);

        const newX = newCenterX - newWidth / 2;
        const newY = newCenterY - newHeight / 2;

        setItems((prev) =>
          prev.map((it) =>
            it.id === isResizing
              ? {
                  ...it,
                  x: newX,
                  y: newY,
                  boxWidth: newWidth,
                  boxHeight: newHeight,
                }
              : it,
          ),
        );
      }
    };

    const handleGlobalUp = () => {
      if (isDragging) {
        setIsDragging(null);
        commitHistory();
      }
      if (isRotating) {
        setIsRotating(null);
        commitHistory();
      }
      if (isResizing) {
        setIsResizing(null);
        setResizeDirection(null);
        commitHistory();
      }
    };

    if (isDragging || isRotating || isResizing) {
      window.addEventListener("pointermove", handleGlobalMove);
      window.addEventListener("pointerup", handleGlobalUp);
    }
    return () => {
      window.removeEventListener("pointermove", handleGlobalMove);
      window.removeEventListener("pointerup", handleGlobalUp);
    };
  }, [
    isDragging,
    dragStart,
    isRotating,
    rotateStart,
    isResizing,
    resizeDirection,
    resizeStart,
    commitHistory,
    selectedIds,
    setItems,
    zoom,
  ]);

  const handlePointerDown = useCallback((e: React.PointerEvent, id: string) => {
    if (activeTool === "hand" || isSpacePanning) {
      return;
    }
    setIsDragging(id);
    if (!selectedIds.includes(id)) {
      if (e.shiftKey) {
        setSelectedIds((prev) => [...prev, id]);
      } else {
        setSelectedIds([id]);
      }
    } else {
      if (e.shiftKey) {
        setSelectedIds((prev) => prev.filter((x) => x !== id));
      }
    }
    setDragStart({ x: e.clientX, y: e.clientY });
    e.preventDefault();
  }, [selectedIds, activeTool, isSpacePanning]);

  const handleResizeStart = useCallback((e: React.PointerEvent, id: string, dir: 'nw' | 'ne' | 'se' | 'sw') => {
    e.preventDefault();
    e.stopPropagation();

    const item = items.find((it) => it.id === id);
    if (!item) return;

    const size = getItemSize(item);

    setIsResizing(id);
    setResizeDirection(dir);
    setResizeStart({
      x: e.clientX,
      y: e.clientY,
      startX: item.x,
      startY: item.y,
      startWidth: size.width,
      startHeight: size.height,
      rotate: item.rotate ?? 0,
    });
  }, [items]);

  const handleRotateStart = useCallback((e: React.PointerEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();

    const el = document.getElementById(`wrapper-${id}`);
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const item = items.find((it) => it.id === id);
    if (!item) return;

    const dx = e.clientX - centerX;
    const dy = e.clientY - centerY;
    const pointerAngle = (Math.atan2(dy, dx) * 180) / Math.PI;
    const startRotation = item.rotate ?? 0;

    setIsRotating(id);
    setRotateStart({
      pointerAngle,
      startRotation,
      centerX,
      centerY,
    });
  }, [items]);

  const handlePaperClick = useCallback((e: React.PointerEvent) => {
    if (activeTool === "hand" || isSpacePanning) {
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      setPanStartOffset({ ...panOffset });
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    if (e.target === e.currentTarget) {
      const canCreate =
        activeTool === "text" ||
        activeTool === "square" ||
        activeTool === "circle" ||
        activeTool === "triangle" ||
        activeTool === "arrow";
      if (canCreate) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - rect.left) / zoom;
        const y = (e.clientY - rect.top) / zoom;

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
        setSelectedIds([]);
        if (activeTool === "pointer") {
          const rect = e.currentTarget.getBoundingClientRect();
          const startX = (e.clientX - rect.left) / zoom;
          const startY = (e.clientY - rect.top) / zoom;
          setMarqueeStart({ x: startX, y: startY });
          setMarqueeEnd({ x: startX, y: startY });
          e.currentTarget.setPointerCapture(e.pointerId);
        }
      }
    }
  }, [
    activeTool,
    isSpacePanning,
    panOffset,
    textColor,
    textSize,
    textLetterSpacing,
    textLineHeight,
    zoom,
    setItems,
    setSelectedId,
    commitHistory,
  ]);

  const handlePaperPointerMove = useCallback((e: React.PointerEvent) => {
    if (isPanning) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      setPanOffset({
        x: panStartOffset.x + dx,
        y: panStartOffset.y + dy,
      });
      return;
    }

    if (marqueeStart) {
      const rect = e.currentTarget.getBoundingClientRect();
      const currentX = (e.clientX - rect.left) / zoom;
      const currentY = (e.clientY - rect.top) / zoom;
      setMarqueeEnd({ x: currentX, y: currentY });
    }
  }, [isPanning, panStart, panStartOffset, marqueeStart, zoom]);

  const handlePaperPointerUp = useCallback((e: React.PointerEvent) => {
    if (isPanning) {
      setIsPanning(false);
      e.currentTarget.releasePointerCapture(e.pointerId);
      return;
    }

    if (marqueeStart && marqueeEnd) {
      e.currentTarget.releasePointerCapture(e.pointerId);

      const x1 = Math.min(marqueeStart.x, marqueeEnd.x);
      const y1 = Math.min(marqueeStart.y, marqueeEnd.y);
      const x2 = Math.max(marqueeStart.x, marqueeEnd.x);
      const y2 = Math.max(marqueeStart.y, marqueeEnd.y);

      if (x2 - x1 > 5 || y2 - y1 > 5) {
        const newlySelected: string[] = [];

        items.forEach((item) => {
          const size = getItemSize(item);
          const itemX = item.x;
          const itemY = item.y;
          const itemW = size.width;
          const itemH = size.height;

          // Check if item bounding box overlaps with marquee box (intersection)
          const intersects = !(
            itemX + itemW < x1 ||
            itemX > x2 ||
            itemY + itemH < y1 ||
            itemY > y2
          );

          if (intersects) {
            newlySelected.push(item.id);
          }
        });

        setSelectedIds(newlySelected);
      } else {
        setSelectedIds([]);
      }

      setMarqueeStart(null);
      setMarqueeEnd(null);
    }
  }, [isPanning, marqueeStart, marqueeEnd, items]);

  const deleteItem = useCallback((id: string) => {
    const idsToDelete = selectedIds.includes(id) ? selectedIds : [id];
    setItems((prev) => prev.filter((it) => !idsToDelete.includes(it.id)));
    setSelectedIds((prev) => prev.filter((x) => !idsToDelete.includes(x)));
    commitHistory();
  }, [selectedIds, setItems, commitHistory]);

  return {
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
  };
}
