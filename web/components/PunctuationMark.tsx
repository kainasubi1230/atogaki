type Props = {
  ch: string;
  color: string;
};

export function PunctuationMark({ ch, color }: Props) {
  if (ch === "。" || ch === "．" || ch === ".") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 18 18"
        style={{
          position: "absolute",
          left: "16%",
          top: "24%",
          width: "62%",
          height: "62%",
          overflow: "visible",
          pointerEvents: "none",
        }}
      >
        <path
          d="M12.4 7.5 C12.0 10.6 9.8 12.7 7.0 12.1 C4.6 11.6 3.5 9.5 4.1 7.1 C4.8 4.7 7.1 3.6 9.4 4.2 C11.4 4.7 12.7 5.9 12.4 7.5"
          fill="none"
          stroke={color}
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (ch === "・") {
    return (
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          left: "35%",
          top: "35%",
          width: "30%",
          height: "30%",
          border: `1.4px solid ${color}`,
          borderRadius: "50%",
          boxSizing: "border-box",
          display: "block",
          pointerEvents: "none",
        }}
      />
    );
  }
  if (ch === "、" || ch === "，" || ch === ",") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 18 28"
        style={{
          position: "absolute",
          left: "26%",
          top: "5%",
          width: "58%",
          height: "86%",
          overflow: "visible",
          pointerEvents: "none",
        }}
      >
        <path
          d="M5.0 4.2 C6.4 10.5 8.8 17.2 13.5 23.2"
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (ch === "！" || ch === "!") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 22 58"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}
      >
        <path d="M12.2 4.5 C11.4 17.0 10.8 29.5 9.4 41.0" fill="none" stroke={color} strokeWidth="2.1" strokeLinecap="round" />
        <path d="M9.5 50.5 C11.8 50.1 13.0 52.4 11.2 54.0 C9.4 55.5 7.2 54.1 7.7 52.1 C8.0 51.0 8.6 50.6 9.5 50.5" fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (ch === "？" || ch === "?") {
    return (
      <svg
        aria-hidden="true"
        viewBox="0 0 34 58"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}
      >
        <path d="M7.2 16.5 C8.8 6.2 25.4 4.7 27.0 16.6 C28.3 26.0 17.3 27.1 16.1 37.4" fill="none" stroke={color} strokeWidth="2.0" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M15.5 50.2 C18.0 49.8 19.1 52.1 17.3 53.9 C15.6 55.5 13.1 54.2 13.7 52.0 C14.0 50.9 14.6 50.4 15.5 50.2" fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return null;
}
