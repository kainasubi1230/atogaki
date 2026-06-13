export type ScreenState = "splash" | "scan" | "canvas";

export type HandwritingPath = {
  id: string;
  d: string;
  strokeWidth: number;
};

export type CanvasItem = {
  id: string;
  type: "text" | "square" | "circle" | "triangle" | "arrow" | "handwriting";
  x: number;
  y: number;
  text: string;
  svg: string;
  color: string;
  isConverted: boolean;
  fontSize?: number;
  letterSpacing?: number;
  lineHeight?: number;
  paths?: HandwritingPath[];
  svgViewBox?: string;
  boxWidth?: number;
  boxHeight?: number;
  rotate?: number;
};
