export function SplashScreen({ onNext }: { onNext: () => void }) {
  return (
    <div
      className="splash-screen animate-fade-in"
      onClick={onNext}
      style={{ cursor: "pointer" }}
    >
      <h1 className="splash-logo">あとがき</h1>
      <p className="splash-subtitle">言葉に、体温を。</p>
      <div className="splash-tech-stack" aria-label="技術ハイライト">
        <span>Dataset-driven glyphs</span>
        <span>Editable SVG strokes</span>
        <span>Safety-gated AI</span>
      </div>
    </div>
  );
}
