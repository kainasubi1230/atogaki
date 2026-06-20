import { useEffect, useState, useCallback } from "react";
import { CanvasItem } from "../lib/types";
import { COLORS } from "../lib/constants";

const INITIAL_PRESENT: CanvasItem[] = [
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
];

export function useCanvasHistory() {
  const [historyState, setHistoryState] = useState({
    past: [] as CanvasItem[][],
    present: INITIAL_PRESENT,
    future: [] as CanvasItem[][],
  });

  const setItems = useCallback(
    (newItemsOrUpdater: CanvasItem[] | ((prev: CanvasItem[]) => CanvasItem[])) => {
      setHistoryState((prev) => {
        const nextPresent =
          typeof newItemsOrUpdater === "function"
            ? newItemsOrUpdater(prev.present)
            : newItemsOrUpdater;
        return { ...prev, present: nextPresent };
      });
    },
    []
  );

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

  const undo = useCallback(() => {
    setHistoryState((prev) => {
      if (prev.past.length === 0) return prev;
      return {
        past: prev.past.slice(0, -1),
        present: prev.past[prev.past.length - 1],
        future: [prev.present, ...prev.future],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setHistoryState((prev) => {
      if (prev.future.length === 0) return prev;
      return {
        past: [...prev.past, prev.present],
        present: prev.future[0],
        future: prev.future.slice(1),
      };
    });
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "z") {
        const activeEl = document.activeElement;
        if (
          activeEl &&
          (activeEl.tagName === "TEXTAREA" ||
            activeEl.tagName === "INPUT" ||
            activeEl.getAttribute("contenteditable") === "true")
        ) {
          return;
        }
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  return {
    items: historyState.present,
    setItems,
    commitHistory,
    undo,
    redo,
    canUndo: historyState.past.length > 0,
    canRedo: historyState.future.length > 0,
  };
}
