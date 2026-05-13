export const COLORS = [
  { id: "black", color: "#2b2b2b", name: "墨黒" },
  { id: "blue", color: "#1c3b57", name: "藍色" },
  { id: "brown", color: "#4a3320", name: "茶褐色" },
];

export type PaperStyle = {
  id: string;
  name: string;
  image?: string;
};

export const PAPERS: PaperStyle[] = [
  { id: "plain", name: "和紙" },
  { id: "lines", name: "便箋" },
  { id: "airmail", name: "エアメール" },
  { id: "genkouyoushi", name: "原稿用紙" },
  { id: "floral", name: "手紙", image: "/papers/floral.png" },
];
