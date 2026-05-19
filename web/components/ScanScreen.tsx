import { useState } from "react";
import { API_BASE } from "../lib/api";

type Props = {
  token: string;
  userId: number | null;
  authStatus: "loading" | "ready" | "error";
  onRetryAuth: () => void;
  onComplete: (styleId: number | null) => void;
};

export function ScanScreen({ token, userId, authStatus, onRetryAuth, onComplete }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState("");
  const isAuthReady = Boolean(token && userId && authStatus === "ready");

  async function handleFileUpload(file: File) {
    setIsProcessing(true);
    setProcessStatus("画像を読み込んでいます...");

    try {
      if (!isAuthReady || !token || !userId) {
        alert("初期化中です。数秒待ってから再試行してください。");
        return;
      }

      const formData = new FormData();
      formData.append("consent", "true");
      formData.append("file", file);

      const uploadRes = await fetch(`${API_BASE}/datasets/upload-scan`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!uploadRes.ok) {
        throw new Error(`upload_failed:${uploadRes.status}`);
      }

      setProcessStatus("前処理を実行しています...");
      const preprocessRes = await fetch(`${API_BASE}/datasets/${userId}/preprocess`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!preprocessRes.ok) {
        throw new Error(`preprocess_failed:${preprocessRes.status}`);
      }

      setProcessStatus("スタイルを準備しています...");
      const trainRes = await fetch(`${API_BASE}/styles/${userId}/train-lora`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!trainRes.ok) {
        throw new Error(`train_failed:${trainRes.status}`);
      }

      const trainBody = (await trainRes.json()) as { style_id?: number };
      if (!trainBody.style_id) {
        throw new Error("style_id_not_returned");
      }

      onComplete(trainBody.style_id);
    } catch (e) {
      console.error(e);
      alert("スタイル準備に失敗しました。API接続と共有スタイル設定を確認してください。");
      onComplete(null);
    } finally {
      setIsProcessing(false);
    }
  }

  const statusLabel =
    authStatus === "ready"
      ? "画像を選択する"
      : authStatus === "error"
        ? "初期化失敗"
        : "初期化中...";

  return (
    <div className="scan-screen animate-fade-in">
      <div className="scan-card animate-slide-up">
        <h2 className="scan-title">手書きサンプルを読み込む</h2>
        <p className="scan-desc">
          紙に書いた文字の画像をアップロードしてください。
          <br />
          スキャン後に前処理とスタイル準備を実行します。
        </p>

        {!isProcessing ? (
          <>
            <div className="upload-btn">
              <span>{statusLabel}</span>
              <input
                type="file"
                accept="image/*"
                disabled={!isAuthReady}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                }}
              />
            </div>
            {authStatus === "error" && (
              <button className="convert-btn" onClick={onRetryAuth} style={{ marginTop: "0.8rem" }}>
                初期化を再試行
              </button>
            )}
          </>
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
