export interface HiddenMotion {
  id: string;
  keywords: string[];
  svg: string; // The SVG string of the silhouette
  className: string;
}

export const HIDDEN_MOTIONS: HiddenMotion[] = [
  {
    id: "car",
    keywords: ["運転", "くるま", "車", "ドライブ", "温水", "運行", "car", "drive"],
    className: "shoji-shadow-car",
    // Silhouette of a retro car (with standard 2 wheels)
    svg: `<svg viewBox="0 0 120 50" width="520" height="216" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M15 30 L5 30 Q1 30 1 25 L1 20 Q1 15 5 15 L20 15 Q25 15 28 10 L38 2 Q40 0 44 0 L76 0 Q80 0 82 2 L92 10 Q95 15 100 15 L115 15 Q119 15 119 20 L119 25 Q119 30 115 30 L105 30 Q105 25 100 25 Q95 25 95 30 L25 30 Q25 25 20 25 Q15 25 15 30 Z" />
      <circle cx="25" cy="32" r="6" />
      <circle cx="95" cy="32" r="6" />
    </svg>`,
  },
  {
    id: "cloud",
    keywords: ["雲", "くも", "クラウド", "雲海", "cloud"],
    className: "shoji-shadow-cloud",
    // Silhouette of a traditional Japanese-style cloud (Kumo)
    svg: `<svg viewBox="0 0 100 45" width="450" height="202" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 35 C15 35 10 32 10 27 C10 22 15 19 20 19 C21 19 22 19 23 19 C26 14 31 10 38 10 C46 10 52 16 53 23 C56 21 60 20 64 20 C71 20 77 25 78 32 C82 31 86 32 88 35 C91 38 90 42 85 43 C80 43 15 43 20 35 Z" />
    </svg>`,
  },
];
