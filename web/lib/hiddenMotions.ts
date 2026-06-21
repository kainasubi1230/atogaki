export interface HiddenMotion {
  id: string;
  keywords: string[];
  svg: string; // The SVG string of the silhouette
  className: string;
}

export const HIDDEN_MOTIONS: HiddenMotion[] = [
  {
    id: "car",
    keywords: ["運転", "くるま", "車", "ドライブ", "温水", "car", "drive"],
    className: "shoji-shadow-car",
    // Silhouette of a retro car (with standard 2 wheels)
    svg: `<svg viewBox="0 0 120 50" width="650" height="270" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
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
    svg: `<svg viewBox="0 0 100 45" width="600" height="270" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 35 C15 35 10 32 10 27 C10 22 15 19 20 19 C21 19 22 19 23 19 C26 14 31 10 38 10 C46 10 52 16 53 23 C56 21 60 20 64 20 C71 20 77 25 78 32 C82 31 86 32 88 35 C91 38 90 42 85 43 C80 43 15 43 20 35 Z" />
    </svg>`,
  },
  {
    id: "airplane",
    keywords: ["運行", "飛行機", "ひこうき", "airplane"],
    className: "shoji-shadow-airplane",
    // Passenger jet plane silhouette
    svg: `<svg viewBox="0 0 100 40" width="550" height="220" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M 10,25 C 10,25 8,12 5,10 C 10,10 14,14 18,22 L 88,22 C 92,22 96,23 96,25 C 96,27 92,28 88,28 L 65,28 L 48,40 L 35,40 L 50,28 L 10,25 Z" />
    </svg>`,
  },
  {
    id: "ship",
    keywords: ["運河", "船", "ふね", "ship"],
    className: "shoji-shadow-ship",
    // Traditional sailboat silhouette
    svg: `<svg viewBox="0 0 100 60" width="450" height="270" fill="currentColor" stroke="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M 10,40 L 90,40 L 80,52 C 75,55 25,55 20,52 Z" stroke="none" />
      <path d="M 48,10 L 48,40 L 50,40 L 50,10 Z" stroke="none" />
      <path d="M 45,12 C 30,20 30,30 45,38 Z" stroke="none" />
      <path d="M 53,15 C 65,22 65,30 53,35 Z" stroke="none" />
      <!-- River / Waves -->
      <path d="M 0,54 Q 15,51 30,54 T 60,54 T 90,54 T 100,54" fill="none" stroke-width="1.8" stroke-linecap="round" />
      <path d="M 5,57 Q 20,55 35,57 T 65,57 T 95,57" fill="none" stroke-width="1.2" stroke-linecap="round" opacity="0.6" />
    </svg>`,
  },
  {
    id: "runner",
    keywords: ["運動", "走る", "ランナー", "runner", "running"],
    className: "shoji-shadow-runner",
    // Hierarchical skeletal jogger silhouette with relative joint rotations
    svg: `<svg viewBox="0 0 100 100" width="320" height="320" fill="currentColor" stroke="currentColor" xmlns="http://www.w3.org/2000/svg">
      <!-- Right Arm (Back Arm) -->
      <g class="runner-upperarm-right">
        <path d="M 52,24 L 40,28" stroke-width="8" stroke-linecap="round" fill="none" />
        <g class="runner-forearm-right">
          <path d="M 40,28 L 44,38" stroke-width="8" stroke-linecap="round" fill="none" />
        </g>
      </g>
      
      <!-- Right Leg (Back Leg) -->
      <g class="runner-thigh-right">
        <path d="M 46,45 L 36,54" stroke-width="9" stroke-linecap="round" fill="none" />
        <g class="runner-calf-right">
          <path d="M 36,54 L 30,66" stroke-width="9" stroke-linecap="round" fill="none" />
        </g>
      </g>
      
      <!-- Left Leg (Front Leg) -->
      <g class="runner-thigh-left">
        <path d="M 46,45 L 54,58" stroke-width="9" stroke-linecap="round" fill="none" />
        <g class="runner-calf-left">
          <path d="M 54,58 L 48,72" stroke-width="9" stroke-linecap="round" fill="none" />
        </g>
      </g>
      
      <!-- Body (Torso and Head) -->
      <path class="runner-body" d="M 52,22 L 46,45" stroke-width="11" stroke-linecap="round" fill="none" />
      <circle cx="54" cy="14" r="6" fill="currentColor" stroke="none" />
      
      <!-- Left Arm (Front Arm) -->
      <g class="runner-upperarm-left">
        <path d="M 52,24 L 62,30" stroke-width="8" stroke-linecap="round" fill="none" />
        <g class="runner-forearm-left">
          <path d="M 62,30 L 56,42" stroke-width="8" stroke-linecap="round" fill="none" />
        </g>
      </g>
    </svg>`,
  },
];
