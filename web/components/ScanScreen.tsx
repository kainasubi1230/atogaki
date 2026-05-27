import { useState } from "react";
import { API_BASE } from "../lib/api";

type Props = {
  token: string;
  userId: number | null;
  authStatus: "loading" | "ready" | "error";
  authError?: string;
  onRetryAuth: () => void;
  onComplete: (styleId: number | null) => void;
};

type JobBody = {
  status?: string;
  error_code?: string | null;
};
type UploadScanBody = {
  dataset_id?: number;
  preprocess_status?: string | null;
  preprocess_error_code?: string | null;
  segment_count?: number | null;
  labeled_segment_count?: number | null;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const SUCCESS_STATUSES = new Set(["completed", "finished", "succeeded", "ready"]);
const FAILURE_STATUSES = new Set(["failed", "error", "cancelled", "canceled"]);
const HIRAGANA_BASIC = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん";
const KATAKANA_BASIC = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン";

function compactChars(text: string): string {
  return Array.from(text)
    .filter((ch) => ch.trim().length > 0)
    .join("");
}

export function ScanScreen({ token, userId, authStatus, authError, onRetryAuth, onComplete }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState("");
  const [sampleLabel, setSampleLabel] = useState("あ");
  const [useBatchLabels, setUseBatchLabels] = useState(true);
  const [batchLabels, setBatchLabels] = useState(`${HIRAGANA_BASIC}${KATAKANA_BASIC}`);
  const isAuthReady = Boolean(token && userId && authStatus === "ready");

  async function waitForJob(tokenValue: string, jobId: string, label: string) {
    let lastStatus = "unknown";
    for (let attempt = 0; attempt < 60; attempt++) {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`, {
        method: "GET",
        headers: { Authorization: `Bearer ${tokenValue}` },
      });
      if (!res.ok) {
        throw new Error(`${label}_job_fetch_failed:${res.status}`);
      }
      const body = (await res.json()) as JobBody;
      const status = body.status ?? "unknown";
      lastStatus = status;
      if (SUCCESS_STATUSES.has(status)) {
        return;
      }
      if (FAILURE_STATUSES.has(status)) {
        throw new Error(`${label}_job_failed:${body.error_code ?? "unknown"}`);
      }
      await sleep(1000);
    }
    throw new Error(`${label}_job_timeout:${lastStatus}`);
  }

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
      const normalizedLabel = sampleLabel.trim();
      const normalizedBatchLabels = compactChars(batchLabels);
      if (useBatchLabels && normalizedBatchLabels.length > 0) {
        formData.append("labels", normalizedBatchLabels);
      } else if (normalizedLabel) {
        formData.append("label", normalizedLabel);
      }

      const uploadRes = await fetch(`${API_BASE}/datasets/upload-scan`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!uploadRes.ok) {
        throw new Error(`upload_failed:${uploadRes.status}`);
      }
      const uploadBody = (await uploadRes.json()) as UploadScanBody;
      const uploadPreprocessStatus = uploadBody.preprocess_status ?? null;

      if (uploadPreprocessStatus === "done") {
        const seg = uploadBody.segment_count ?? 0;
        const labeled = uploadBody.labeled_segment_count ?? 0;
        if (seg > 0) {
          setProcessStatus(`ラベル付き前処理が完了しました（切り出し ${seg} / ラベル付与 ${labeled}）。`);
        } else {
          setProcessStatus("ラベル付き前処理が完了しました。");
        }
      } else if (uploadPreprocessStatus === "failed") {
        throw new Error(`preprocess_failed:upload:${uploadBody.preprocess_error_code ?? "unknown"}`);
      } else {
        setProcessStatus("前処理を実行しています...");
        const preprocessRes = await fetch(`${API_BASE}/datasets/${userId}/preprocess`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!preprocessRes.ok) {
          throw new Error(`preprocess_failed:${preprocessRes.status}`);
        }
        const preprocessBody = (await preprocessRes.json()) as { job_id?: string };
        if (!preprocessBody.job_id) {
          throw new Error("preprocess_job_id_not_returned");
        }

        setProcessStatus("前処理ジョブの完了を待っています...");
        await waitForJob(token, preprocessBody.job_id, "preprocess");
      }

      setProcessStatus("スタイルを準備しています...");
      const trainRes = await fetch(`${API_BASE}/styles/${userId}/train-lora`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!trainRes.ok) {
        throw new Error(`train_failed:${trainRes.status}`);
      }

      const trainBody = (await trainRes.json()) as {
        style_id?: number;
        job_id?: string;
        status?: string;
      };
      if (!trainBody.style_id) {
        throw new Error("style_id_not_returned");
      }

      if (trainBody.status !== "ready") {
        if (!trainBody.job_id) {
          throw new Error("train_job_id_not_returned");
        }
        setProcessStatus("スタイル学習ジョブの完了を待っています...");
        await waitForJob(token, trainBody.job_id, "train_lora");
      }

      onComplete(trainBody.style_id);
    } catch (e) {
      console.error(e);
      const message = e instanceof Error ? e.message : String(e);
      if (message.includes("_job_timeout:queued")) {
        alert("ジョブがキュー待機のままです。workerプロセス（default/trainer）が起動しているか確認してください。");
      } else if (message.includes("preprocess_failed:upload:LOW_CONTRAST")) {
        alert("画像のコントラストが低すぎます。紙を濃く書いて、明るい場所で撮影してください。");
      } else if (message.includes("preprocess_failed:upload:NO_TEXT_DETECTED")) {
        alert("文字を検出できませんでした。背景を単純にして、紙が画面いっぱいに入るように撮影してください。");
      } else if (message.includes("preprocess_failed:upload:TOO_FEW_SEGMENTS")) {
        alert("文字の切り出しに失敗しました。画像の明るさ・コントラストを上げ、文字同士の間隔を広めにして再試行してください。");
      } else {
        alert("スタイル準備に失敗しました。前処理/学習ジョブが失敗していないか確認してください。");
      }
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
            <p style={{ marginTop: "0.5rem", fontSize: "0.8rem", opacity: 0.88 }}>
              1枚にひらがな・カタカナを複数書いた画像は「一括ラベル」をONにしてアップロードしてください（左上から読み順で対応）。
            </p>
            {authStatus === "error" && authError ? (
              <p style={{ marginTop: "0.5rem", fontSize: "0.8rem", color: "#ffd8d8" }}>
                初期化エラー: {authError}
              </p>
            ) : null}
            <div
              style={{
                marginTop: "0.8rem",
                textAlign: "left",
                color: "rgba(255,255,255,0.94)",
              }}
            >
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.86rem" }}>
                <input
                  type="checkbox"
                  checked={useBatchLabels}
                  disabled={!isAuthReady || isProcessing}
                  onChange={(e) => setUseBatchLabels(e.target.checked)}
                />
                かな一括ラベルを使う（1枚画像向け）
              </label>
              <textarea
                value={batchLabels}
                disabled={!isAuthReady || isProcessing || !useBatchLabels}
                onChange={(e) => setBatchLabels(e.target.value)}
                rows={3}
                style={{
                  marginTop: "0.45rem",
                  width: "100%",
                  borderRadius: "8px",
                  border: "1px solid rgba(255,255,255,0.35)",
                  background: useBatchLabels ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.08)",
                  color: "white",
                  padding: "0.45rem 0.55rem",
                  fontSize: "0.82rem",
                  lineHeight: 1.3,
                }}
              />
              <p style={{ margin: "0.35rem 0 0", fontSize: "0.74rem", opacity: 0.8 }}>
                現在 {compactChars(batchLabels).length} 文字。推奨: ひらがな→カタカナ順で画像上に並べる。
              </p>
              {!useBatchLabels ? (
                <div style={{ marginTop: "0.6rem", display: "flex", gap: "0.6rem", alignItems: "center" }}>
                  <label style={{ fontSize: "0.84rem", minWidth: "3.2rem" }}>単一ラベル</label>
                  <input
                    type="text"
                    value={sampleLabel}
                    maxLength={64}
                    disabled={!isAuthReady || isProcessing}
                    onChange={(e) => setSampleLabel(e.target.value)}
                    style={{
                      flex: 1,
                      borderRadius: "8px",
                      border: "1px solid rgba(255,255,255,0.35)",
                      background: "rgba(255,255,255,0.15)",
                      color: "white",
                      padding: "0.4rem 0.55rem",
                      fontSize: "0.86rem",
                    }}
                  />
                </div>
              ) : null}
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
