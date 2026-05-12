import { useState } from "react";
import { API_BASE } from "../lib/api";

type Props = {
  token: string;
  userId: number | null;
  onComplete: (styleId: number | null) => void;
};

export function ScanScreen({ token, userId, onComplete }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState("");

  async function handleFileUpload(file: File) {
    setIsProcessing(true);
    setProcessStatus("筆跡を読み込んでいます...");

    try {
      if (token && userId) {
        const formData = new FormData();
        formData.append("consent", "true");
        formData.append("file", file);
        await fetch(`${API_BASE}/datasets/upload-scan`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        });

        setProcessStatus("文字の特徴を抽出しています...");
        await fetch(`${API_BASE}/datasets/${userId}/preprocess`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });

        setProcessStatus("専用フォントを作成しています...");
        const trainRes = await fetch(
          `${API_BASE}/styles/${userId}/train-lora`,
          {
            method: "POST",
            headers: { Authorization: `Bearer ${token}` },
          },
        );

        if (trainRes.status === 200) {
          const trainBody = await trainRes.json();
          onComplete(trainBody.style_id);
          return;
        }
      } else {
        // Fallback delay for prototype showcase if backend is down
        await new Promise((r) => setTimeout(r, 2000));
      }
      onComplete(null);
    } catch (e) {
      console.error(e);
      onComplete(null);
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <div className="scan-screen animate-fade-in">
      <div className="scan-card animate-slide-up">
        <h2 className="scan-title">あなたの字を、教える。</h2>
        <p className="scan-desc">
          紙に書いた文字を写真に撮ってアップロードしてください。
          <br />
          あなた専用の「手書きフォント」を作成します。
        </p>

        {!isProcessing ? (
          <div className="upload-btn">
            <span>手書きをスキャンする</span>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </div>
        ) : (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>{processStatus}</p>
          </div>
        )}
      </div>
    </div>
  );
}
