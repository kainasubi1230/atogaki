import { useState } from "react";
import { API_BASE } from "../lib/api";

type Props = {
  token: string;
  userId: number | null;
  authStatus: "loading" | "ready" | "error";
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
};

type TrajectoryPoint = {
  x: number;
  y: number;
  t: number;
  pen_state: "down" | "up";
  width: number;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const SUCCESS_STATUSES = new Set(["completed", "finished", "succeeded", "ready"]);
const FAILURE_STATUSES = new Set(["failed", "error", "cancelled", "canceled"]);

export function ScanScreen({ token, userId, authStatus, onRetryAuth, onComplete }: Props) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState("");
  const [isSavingTrajectory, setIsSavingTrajectory] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [trajectoryPoints, setTrajectoryPoints] = useState<TrajectoryPoint[]>([]);
  const [tick, setTick] = useState(0);
  const [sampleLabel, setSampleLabel] = useState("あ");
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
      if (normalizedLabel) {
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
        setProcessStatus("ラベル付き前処理が完了しました。");
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
        alert("文字の切り出しに失敗しました。1文字を大きく書いた画像で再試行してください。");
      } else {
        alert("スタイル準備に失敗しました。前処理/学習ジョブが失敗していないか確認してください。");
      }
      onComplete(null);
    } finally {
      setIsProcessing(false);
    }
  }

  function pointFromEvent(
    e: React.PointerEvent<HTMLDivElement>,
  ): { x: number; y: number } {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(rect.width, e.clientX - rect.left)),
      y: Math.max(0, Math.min(rect.height, e.clientY - rect.top)),
    };
  }

  function handlePadPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (!isAuthReady || isSavingTrajectory || isProcessing) return;
    const p = pointFromEvent(e);
    setIsDrawing(true);
    setTrajectoryPoints((prev) => [
      ...prev,
      { x: p.x, y: p.y, t: tick, pen_state: "down", width: 2.0 },
    ]);
    setTick((prev) => prev + 1);
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function handlePadPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDrawing || !isAuthReady || isSavingTrajectory || isProcessing) return;
    const p = pointFromEvent(e);
    setTrajectoryPoints((prev) => [
      ...prev,
      { x: p.x, y: p.y, t: tick, pen_state: "down", width: 2.0 },
    ]);
    setTick((prev) => prev + 1);
  }

  function handlePadPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDrawing || !isAuthReady || isSavingTrajectory || isProcessing) return;
    const p = pointFromEvent(e);
    setTrajectoryPoints((prev) => [
      ...prev,
      { x: p.x, y: p.y, t: tick, pen_state: "up", width: 2.0 },
    ]);
    setTick((prev) => prev + 1);
    setIsDrawing(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  }

  function clearTrajectoryPad() {
    setTrajectoryPoints([]);
    setTick(0);
    setIsDrawing(false);
  }

  async function saveTrajectorySample() {
    if (!isAuthReady || !token || !userId) {
      alert("初期化中です。数秒待ってから再試行してください。");
      return;
    }
    if (trajectoryPoints.length < 2) {
      alert("先に手書きサンプルを描いてください。");
      return;
    }
    const normalizedLabel = sampleLabel.trim();
    if (!normalizedLabel) {
      alert("ラベルは必須です（例: あ）。");
      return;
    }

    setIsSavingTrajectory(true);
    try {
      const res = await fetch(`${API_BASE}/datasets/upload-trajectory`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          consent: true,
          label: normalizedLabel,
          points: trajectoryPoints,
        }),
      });
      if (!res.ok) {
        let detail = "";
        try {
          const err = (await res.json()) as { detail?: string };
          detail = err.detail ?? "";
        } catch {
          // ignore parse errors and fall back to status only
        }
        throw new Error(`trajectory_upload_failed:${res.status}:${detail}`);
      }
      const body = (await res.json()) as { dataset_id?: number; point_count?: number };
      alert(
        `軌跡サンプルを保存しました（dataset_id=${body.dataset_id ?? "?"}, points=${body.point_count ?? trajectoryPoints.length}）`,
      );
      clearTrajectoryPad();
    } catch (e) {
      console.error(e);
      const message = e instanceof Error ? e.message : String(e);
      if (message.includes("trajectory_too_short")) {
        alert("軌跡が短すぎます。1文字をゆっくり、はっきり描いてください。");
      } else if (message.includes("trajectory_too_small")) {
        alert("描画サイズが小さすぎます。もう少し大きく書いてください。");
      } else if (message.includes("label_required")) {
        alert("ラベルを入力してください（例: あ）。");
      } else {
        alert("軌跡サンプルの保存に失敗しました。");
      }
    } finally {
      setIsSavingTrajectory(false);
    }
  }

  const trajectoryPath = trajectoryPoints
    .map((p, idx) => `${idx === 0 || p.pen_state !== "down" ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");

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
              紙画像を学習データ化する場合は、下のラベル欄に文字（例: あ）を入れてからアップロードしてください。
            </p>
            {authStatus === "error" && (
              <button className="convert-btn" onClick={onRetryAuth} style={{ marginTop: "0.8rem" }}>
                初期化を再試行
              </button>
            )}

            <div
              style={{
                marginTop: "1.4rem",
                textAlign: "left",
                color: "rgba(255,255,255,0.94)",
              }}
            >
              <p style={{ margin: "0 0 0.5rem", fontSize: "0.92rem", opacity: 0.95 }}>
                Web手書きサンプル（学習データ用）
              </p>
              <div
                style={{
                  display: "flex",
                  gap: "0.6rem",
                  alignItems: "center",
                  marginBottom: "0.55rem",
                }}
              >
                <label style={{ fontSize: "0.85rem", minWidth: "3.2rem" }}>ラベル</label>
                <input
                  type="text"
                  value={sampleLabel}
                  maxLength={64}
                  disabled={!isAuthReady || isSavingTrajectory}
                  onChange={(e) => setSampleLabel(e.target.value)}
                  style={{
                    flex: 1,
                    borderRadius: "8px",
                    border: "1px solid rgba(255,255,255,0.35)",
                    background: "rgba(255,255,255,0.15)",
                    color: "white",
                    padding: "0.4rem 0.55rem",
                    fontSize: "0.9rem",
                  }}
                />
              </div>
              <div
                role="presentation"
                onPointerDown={handlePadPointerDown}
                onPointerMove={handlePadPointerMove}
                onPointerUp={handlePadPointerUp}
                onPointerCancel={handlePadPointerUp}
                style={{
                  width: "100%",
                  height: "180px",
                  borderRadius: "10px",
                  border: "1px solid rgba(255,255,255,0.35)",
                  background: "rgba(255,255,255,0.08)",
                  position: "relative",
                  cursor: isAuthReady ? "crosshair" : "not-allowed",
                  overflow: "hidden",
                }}
              >
                <svg width="100%" height="100%">
                  {trajectoryPath ? (
                    <path
                      d={trajectoryPath}
                      stroke="white"
                      strokeWidth={2}
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      vectorEffect="non-scaling-stroke"
                    />
                  ) : null}
                </svg>
              </div>
              <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.6rem" }}>
                <button
                  className="convert-btn"
                  disabled={!isAuthReady || isSavingTrajectory || trajectoryPoints.length === 0}
                  onClick={clearTrajectoryPad}
                  style={{ flex: 1 }}
                >
                  クリア
                </button>
                <button
                  className="convert-btn"
                  disabled={!isAuthReady || isSavingTrajectory || trajectoryPoints.length < 2 || sampleLabel.trim().length === 0}
                  onClick={saveTrajectorySample}
                  style={{ flex: 1 }}
                >
                  {isSavingTrajectory ? "保存中..." : "軌跡を保存"}
                </button>
              </div>
              <p style={{ margin: "0.45rem 0 0", fontSize: "0.78rem", opacity: 0.86 }}>
                ラベル付きで保存すると前処理済みデータとして登録され、ベース学習データに直接使えます。
              </p>
            </div>
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
