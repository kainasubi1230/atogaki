"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type ApiResult = { status: number; body: unknown };

async function postJSON(path: string, body: unknown, token?: string): Promise<ApiResult> {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      },
      body: JSON.stringify(body)
    });
    return { status: response.status, body: await response.json() };
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown_error";
    return {
      status: 0,
      body: {
        detail: `network_error: ${message}`
      }
    };
  }
}

type ScreenState = 'splash' | 'scan' | 'canvas';

export default function Page() {
  const [screen, setScreen] = useState<ScreenState>('splash');
  const [token, setToken] = useState("");
  const [userId, setUserId] = useState<number | null>(null);
  const [styleId, setStyleId] = useState<number | null>(null);

  const [text, setText] = useState("");
  const [svg, setSvg] = useState("");
  const [isConverted, setIsConverted] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState("");

  useEffect(() => {
    // 1. Auto Signup for seamless UX
    const initApp = async () => {
      const email = `guest_${Date.now()}@example.com`;
      const pwd = "password";
      try {
        const res = await postJSON("/auth/signup", { email, password: pwd });
        if (res.status === 200) {
          const body = res.body as { access_token: string; user_id: number };
          setToken(body.access_token);
          setUserId(body.user_id);
        }
      } catch (e) {
        console.error("Auto signup failed", e);
      }

      // 2. Wait a minimum of 2.5 seconds to show the beautiful splash
      setTimeout(() => {
        setScreen('scan');
      }, 2500);
    };
    
    initApp();
  }, []);

  async function handleFileUpload(file: File) {
    setIsProcessing(true);
    setProcessStatus("筆跡を読み込んでいます...");
    
    try {
      if (token && userId) {
        // Upload
        const formData = new FormData();
        formData.append("consent", "true");
        formData.append("file", file);
        await fetch(`${API_BASE}/datasets/upload-scan`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: formData
        });

        setProcessStatus("文字の特徴を抽出しています...");
        // Preprocess
        await fetch(`${API_BASE}/datasets/${userId}/preprocess`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` }
        });

        setProcessStatus("専用フォントを作成しています...");
        // Train
        const trainRes = await fetch(`${API_BASE}/styles/${userId}/train-lora`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (trainRes.status === 200) {
          const trainBody = await trainRes.json();
          setStyleId(trainBody.style_id);
        }
      } else {
        // Fallback delay for prototype showcase if backend is down
        await new Promise(r => setTimeout(r, 2000));
      }
      
      // Move to canvas
      setScreen('canvas');
    } catch (e) {
      console.error(e);
      // Even if backend fails, proceed to canvas to show UX
      setScreen('canvas');
    } finally {
      setIsProcessing(false);
    }
  }

  async function handleConvert() {
    if (!text.trim()) return;
    setIsProcessing(true);
    try {
      if (token && userId && styleId) {
        const result = await postJSON(
          "/generate",
          { user_id: userId, style_id: styleId, text, purpose: "accessibility" },
          token
        );
        if (result.status === 200) {
          const body = result.body as { svg: string };
          setSvg(body.svg);
        }
      } else {
        // Fallback for prototype showcase without backend
        // Ideally we'd show a dummy SVG here, but for now we just fade out text
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsConverted(true);
      setIsProcessing(false);
    }
  }

  return (
    <div className="app-container">
      {screen === 'splash' && (
        <div className="splash-screen animate-fade-in">
          <h1 className="splash-logo">あとがき</h1>
          <p className="splash-subtitle">言葉に、体温を。</p>
        </div>
      )}

      {screen === 'scan' && (
        <div className="scan-screen animate-fade-in">
          <div className="scan-card animate-slide-up">
            <h2 className="scan-title">あなたの字を、教える。</h2>
            <p className="scan-desc">
              紙に書いた文字を写真に撮ってアップロードしてください。<br/>
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
      )}

      {screen === 'canvas' && (
        <div className="canvas-screen animate-fade-in">
          <div className="paper animate-slide-up">
            <textarea
              className={`letter-input ${isConverted ? 'hidden-text' : ''}`}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="ここにメッセージを入力してください..."
              disabled={isConverted || isProcessing}
            />
            {svg && (
              <div 
                className={`handwriting-svg ${isConverted ? 'visible' : ''}`} 
                dangerouslySetInnerHTML={{ __html: svg }} 
              />
            )}
          </div>
          
          <div className="convert-btn-wrapper animate-fade-in" style={{ animationDelay: '0.4s', animationFillMode: 'both' }}>
            {!isConverted && (
              <button 
                className="convert-btn" 
                onClick={handleConvert}
                disabled={!text.trim() || isProcessing}
              >
                {isProcessing ? "変換中..." : "筆跡に変換する"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
