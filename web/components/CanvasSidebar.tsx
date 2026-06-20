import React from "react";
import { COLORS, PAPERS } from "../lib/constants";

type Props = {
  isMenuOpen: boolean;
  setIsMenuOpen: (open: boolean) => void;
  activeTab: "paper" | "write";
  setActiveTab: (tab: "paper" | "write") => void;
  paperStyle: string;
  setPaperStyle: (style: string) => void;
  orientation: "portrait" | "landscape";
  setOrientation: (orientation: "portrait" | "landscape") => void;
  activeTool: "pointer" | "text" | "square" | "circle" | "triangle" | "arrow" | "hand";
  setActiveTool: (tool: "pointer" | "text" | "square" | "circle" | "triangle" | "arrow" | "hand") => void;
  currentFontSize: number;
  handleFontSizeChange: (size: number) => void;
  currentLetterSpacing: number;
  handleLetterSpacingChange: (spacing: number) => void;
  currentLineHeight: number;
  handleLineHeightChange: (height: number) => void;
  commitHistory: () => void;
  currentColor: string;
  handleColorChange: (color: string) => void;
  customColor: string;
  setCustomColor: (color: string) => void;
  isCustomColorActive: boolean;
  customColorIconColor: string;
  savedColors: string[];
  saveColor: (color: string) => void;
  deleteSavedColor: (color: string) => void;
};

export function CanvasSidebar({
  isMenuOpen,
  setIsMenuOpen,
  activeTab,
  setActiveTab,
  paperStyle,
  setPaperStyle,
  orientation,
  setOrientation,
  activeTool,
  setActiveTool,
  currentFontSize,
  handleFontSizeChange,
  currentLetterSpacing,
  handleLetterSpacingChange,
  currentLineHeight,
  handleLineHeightChange,
  commitHistory,
  currentColor,
  handleColorChange,
  customColor,
  setCustomColor,
  isCustomColorActive,
  customColorIconColor,
  savedColors,
  saveColor,
  deleteSavedColor,
}: Props) {
  return (
    <>
      <button
        className={`menu-toggle-btn ${!isMenuOpen ? "closed" : ""}`}
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        title={isMenuOpen ? "メニューを閉じる" : "メニューを開く"}
      >
        {isMenuOpen ? (
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6"></polyline>
          </svg>
        ) : (
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
        )}
      </button>

      <div className={`sidebar ${!isMenuOpen ? "closed" : ""} animate-slide-up`}>
        <div className="sidebar-tabs">
          <button
            className={`sidebar-tab-btn ${activeTab === "paper" ? "active" : ""}`}
            onClick={() => setActiveTab("paper")}
          >
            便箋デザイン
          </button>
          <button
            className={`sidebar-tab-btn ${activeTab === "write" ? "active" : ""}`}
            onClick={() => setActiveTab("write")}
          >
            文字・ツール
          </button>
        </div>

        {activeTab === "paper" && (
          <div className="sidebar-section">
            <h3>便箋デザイン</h3>

            <div style={{ marginBottom: "1rem" }}>
              <div
                style={{
                  fontSize: "0.85rem",
                  marginBottom: "0.4rem",
                  color: "var(--text-secondary)",
                }}
              >
                種類
              </div>
              <div className="paper-options">
                {PAPERS.map((p) => (
                  <button
                    key={p.id}
                    className={`paper-btn ${paperStyle === p.id ? "active" : ""}`}
                    onClick={() => setPaperStyle(p.id)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div
                style={{
                  fontSize: "0.85rem",
                  marginBottom: "0.4rem",
                  color: "var(--text-secondary)",
                }}
              >
                向き
              </div>
              <div className="tool-grid">
                <button
                  className={`tool-btn ${orientation === "portrait" ? "active" : ""}`}
                  onClick={() => setOrientation("portrait")}
                  style={{ padding: "0.6rem 0" }}
                >
                  縦向き
                </button>
                <button
                  className={`tool-btn ${orientation === "landscape" ? "active" : ""}`}
                  onClick={() => setOrientation("landscape")}
                  style={{ padding: "0.6rem 0" }}
                >
                  横向き
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === "write" && (
          <>
            <div className="sidebar-section">
              <h3>文字を書く</h3>
              <div className="tool-grid">
                <button
                  className={`tool-btn ${activeTool === "pointer" ? "active" : ""}`}
                  onClick={() => setActiveTool("pointer")}
                  title="選択・移動"
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
                    <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                    <path d="M13 13l6 6"></path>
                  </svg>
                  <span>選択</span>
                </button>
                <button
                  className={`tool-btn ${activeTool === "hand" ? "active" : ""}`}
                  onClick={() => setActiveTool("hand")}
                  title="手のひらツール（キャンバスの移動）"
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
                    <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v5"></path>
                    <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v6"></path>
                    <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v4.5"></path>
                    <path d="M6 10a2 2 0 0 0-2 2v5a7 7 0 0 0 7 7h1a7 7 0 0 0 7-7v-6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2"></path>
                  </svg>
                  <span>手のひら</span>
                </button>
                <button
                  className={`tool-btn ${activeTool === "text" ? "active" : ""}`}
                  onClick={() => setActiveTool("text")}
                  title="テキスト"
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
                    <polyline points="4 7 4 4 20 4 20 7"></polyline>
                    <line x1="9" y1="20" x2="15" y2="20"></line>
                    <line x1="12" y1="4" x2="12" y2="20"></line>
                  </svg>
                  <span>文字</span>
                </button>
              </div>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <h3>図形</h3>
              <div className="tool-grid">
                <button
                  className={`tool-btn ${activeTool === "square" ? "active" : ""}`}
                  onClick={() => setActiveTool("square")}
                  title="四角形"
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
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  </svg>
                  <span>四角</span>
                </button>
                <button
                  className={`tool-btn ${activeTool === "circle" ? "active" : ""}`}
                  onClick={() => setActiveTool("circle")}
                  title="円形"
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
                    <circle cx="12" cy="12" r="10"></circle>
                  </svg>
                  <span>丸</span>
                </button>
                <button
                  className={`tool-btn ${activeTool === "triangle" ? "active" : ""}`}
                  onClick={() => setActiveTool("triangle")}
                  title="三角形"
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
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                  </svg>
                  <span>三角</span>
                </button>
                <button
                  className={`tool-btn ${activeTool === "arrow" ? "active" : ""}`}
                  onClick={() => setActiveTool("arrow")}
                  title="矢印"
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
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                  </svg>
                  <span>矢印</span>
                </button>
              </div>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <h3>文字のスタイル</h3>

              <div>
                <div
                  style={{
                    fontSize: "0.85rem",
                    marginBottom: "0.4rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  文字の大きさ
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>小</span>
                  <input
                    type="range"
                    min="14"
                    max="60"
                    value={currentFontSize}
                    onChange={(e) => handleFontSizeChange(parseInt(e.target.value))}
                    onPointerUp={commitHistory}
                    className="sidebar-slider"
                  />
                  <span style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}>大</span>
                </div>
              </div>

              <div style={{ marginTop: "0.75rem" }}>
                <div
                  style={{
                    fontSize: "0.85rem",
                    marginBottom: "0.4rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  文字の間隔（横）
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>狭</span>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="1"
                    value={currentLetterSpacing}
                    onChange={(e) => handleLetterSpacingChange(parseInt(e.target.value))}
                    onPointerUp={commitHistory}
                    className="sidebar-slider"
                  />
                  <span style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}>広</span>
                </div>
              </div>

              <div style={{ marginTop: "0.75rem" }}>
                <div
                  style={{
                    fontSize: "0.85rem",
                    marginBottom: "0.4rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  行の間隔（縦）
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>狭</span>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    step="0.1"
                    value={currentLineHeight}
                    onChange={(e) => handleLineHeightChange(parseFloat(e.target.value))}
                    onPointerUp={commitHistory}
                    className="sidebar-slider"
                  />
                  <span style={{ fontSize: "1.2rem", color: "var(--text-secondary)" }}>広</span>
                </div>
              </div>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <h3>インクの色</h3>
              <div className="color-options" style={{ alignItems: "center" }}>
                {COLORS.map((c) => (
                  <button
                    key={c.id}
                    className={`color-btn ${currentColor === c.color ? "active" : ""}`}
                    style={{ backgroundColor: c.color }}
                    title={c.name}
                    onClick={() => {
                      handleColorChange(c.color);
                      commitHistory();
                    }}
                  />
                ))}
                <div
                  className={`color-picker-wrapper ${isCustomColorActive ? "active" : ""}`}
                  title="自由な色を選ぶ"
                  style={{
                    backgroundColor: isCustomColorActive ? currentColor : "#fcfaf7",
                    borderStyle: isCustomColorActive ? "solid" : "dashed",
                  }}
                  onClick={() => {
                    if (!isCustomColorActive) {
                      handleColorChange(customColor);
                      commitHistory();
                    }
                  }}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke={customColorIconColor}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{
                      position: "absolute",
                      zIndex: 2,
                      pointerEvents: "none",
                      filter: isCustomColorActive
                        ? "drop-shadow(0px 1px 1px rgba(0,0,0,0.3))"
                        : "none",
                    }}
                  >
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 14.7255 3.09032 17.1962 4.85857 19C5.02845 19.17 5.11339 19.255 5.21855 19.3005C5.32371 19.346 5.48316 19.346 5.80206 19.346H7C7.55228 19.346 8 18.8983 8 18.346C8 17.7937 8.44772 17.346 9 17.346H11C11.5523 17.346 12 16.8983 12 16.346V15C12 14.4477 12.4477 14 13 14H15C16.1046 14 17 13.1046 17 12V11C17 9.89543 16.1046 9 15 9H9C7.89543 9 7 9.89543 7 11V14.346" />
                  </svg>
                  <input
                    type="color"
                    className="color-picker-input"
                    style={{ pointerEvents: isCustomColorActive ? "auto" : "none" }}
                    value={customColor}
                    onChange={(e) => {
                      const val = e.target.value;
                      setCustomColor(val);
                      localStorage.setItem("machine_custom_color_v1", val);
                      handleColorChange(val);
                    }}
                    onBlur={commitHistory}
                  />
                </div>
              </div>
              {isCustomColorActive && !savedColors.includes(currentColor) && (
                <div style={{ width: "100%", marginTop: "0.6rem" }}>
                  <button
                    onClick={() => saveColor(currentColor)}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      background: "transparent",
                      border: "1px dashed rgba(0, 0, 0, 0.25)",
                      borderRadius: "4px",
                      padding: "6px 10px",
                      fontSize: "0.78rem",
                      color: "var(--text-secondary)",
                      cursor: "pointer",
                      fontFamily: "inherit",
                      transition: "all 0.2s",
                      backgroundColor: "#fcfaf7",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "rgba(0,0,0,0.4)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "rgba(0,0,0,0.25)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
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
                      <line x1="12" y1="5" x2="12" y2="19"></line>
                      <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    <span>この色をお気に入りに保存</span>
                  </button>
                </div>
              )}

              {savedColors.length > 0 && (
                <div
                  style={{
                    display: "flex",
                    gap: "0.8rem",
                    flexWrap: "wrap",
                    marginTop: "0.8rem",
                    width: "100%",
                    borderTop: "1px dotted rgba(0, 0, 0, 0.15)",
                    paddingTop: "0.8rem",
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-secondary)",
                      width: "100%",
                      marginBottom: "-0.4rem",
                    }}
                  >
                    マイインク（保存した色）
                  </div>
                  {savedColors.map((color) => (
                    <div key={color} className="saved-color-container">
                      <button
                        className={`color-btn ${currentColor === color ? "active" : ""}`}
                        style={{ backgroundColor: color }}
                        onClick={() => {
                          handleColorChange(color);
                          commitHistory();
                        }}
                      />
                      <button
                        className="delete-color-btn"
                        title="削除"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSavedColor(color);
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}
