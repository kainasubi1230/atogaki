import { HIDDEN_MOTIONS } from "../lib/hiddenMotions";

type Props = {
  isProcessing: boolean;
  activeMotions?: string[];
};

export function ShojiOverlay({ isProcessing, activeMotions = [] }: Props) {
  return (
    <div className={`shoji-overlay ${isProcessing ? "active" : ""} ${activeMotions.length > 0 ? "has-shadow-motions" : ""}`}>
      {/* Shadows rendered behind the doors */}
      {isProcessing && activeMotions.map((motionId) => {
        const motion = HIDDEN_MOTIONS.find(m => m.id === motionId);
        if (!motion) return null;
        return (
          <div
            key={motion.id}
            className={`shoji-shadow-silhouette ${motion.className}`}
            dangerouslySetInnerHTML={{ __html: motion.svg }}
          />
        );
      })}

      <div className="shoji-door left">
        <div className="shoji-grid">
          {Array.from({ length: 24 }).map((_, i) => (
            <div key={i} className="shoji-cell" />
          ))}
        </div>
        <div className="shoji-hikite">
          <div className="shoji-hikite-inner" />
        </div>
      </div>
      <div className="shoji-door right">
        <div className="shoji-grid">
          {Array.from({ length: 24 }).map((_, i) => (
            <div key={i} className="shoji-cell" />
          ))}
        </div>
        <div className="shoji-hikite">
          <div className="shoji-hikite-inner" />
        </div>
      </div>
      <div className="shoji-content">
        <div className="shoji-text">筆跡構築中</div>
      </div>
    </div>
  );
}
