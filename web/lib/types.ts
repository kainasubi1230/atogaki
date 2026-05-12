export type ScreenState = "splash" | "scan" | "canvas";

export type CanvasItem = {
  id: string;
  type: "text" | "square" | "circle" | "triangle";
  x: number;
  y: number;
  text: string;
  svg: string;
  color: string;
  isConverted: boolean;
  fontSize?: number;
};
