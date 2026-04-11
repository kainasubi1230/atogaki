"use client";

import { FormEvent, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type ApiResult = { status: number; body: unknown };

async function postJSON(path: string, body: unknown, token?: string): Promise<ApiResult> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify(body)
  });
  return { status: response.status, body: await response.json() };
}

export default function Page() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [userId, setUserId] = useState<number | null>(null);
  const [styleId, setStyleId] = useState<number | null>(null);
  const [text, setText] = useState("こんにちは、支援モードです。");
  const [log, setLog] = useState<string>("Ready.");
  const [svg, setSvg] = useState("");

  const authReady = useMemo(() => !!token && userId !== null, [token, userId]);

  async function onSignup(e: FormEvent) {
    e.preventDefault();
    const result = await postJSON("/auth/signup", { email, password });
    if (result.status === 200) {
      const body = result.body as { access_token: string; user_id: number };
      setToken(body.access_token);
      setUserId(body.user_id);
      setLog(`signed up user_id=${body.user_id}`);
      return;
    }
    setLog(JSON.stringify(result.body));
  }

  async function onLogin(e: FormEvent) {
    e.preventDefault();
    const result = await postJSON("/auth/login", { email, password });
    if (result.status === 200) {
      const body = result.body as { access_token: string; user_id: number };
      setToken(body.access_token);
      setUserId(body.user_id);
      setLog(`logged in user_id=${body.user_id}`);
      return;
    }
    setLog(JSON.stringify(result.body));
  }

  async function onUpload(file: File) {
    if (!authReady || userId === null) return;
    const formData = new FormData();
    formData.append("consent", "true");
    formData.append("file", file);
    const response = await fetch(`${API_BASE}/datasets/upload-scan`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData
    });
    const body = await response.json();
    setLog(`upload: ${JSON.stringify(body)}`);
  }

  async function onPreprocess() {
    if (!authReady || userId === null) return;
    const response = await fetch(`${API_BASE}/datasets/${userId}/preprocess`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` }
    });
    const body = await response.json();
    setLog(`preprocess: ${JSON.stringify(body)}`);
  }

  async function onTrain() {
    if (!authReady || userId === null) return;
    const response = await fetch(`${API_BASE}/styles/${userId}/train-lora`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` }
    });
    const body = await response.json();
    if (response.status === 200) setStyleId(body.style_id);
    setLog(`train: ${JSON.stringify(body)}`);
  }

  async function onGenerate() {
    if (!authReady || userId === null || styleId === null) return;
    const result = await postJSON(
      "/generate",
      { user_id: userId, style_id: styleId, text, purpose: "accessibility" },
      token
    );
    if (result.status === 200) {
      const body = result.body as { output_id: number; svg: string };
      setSvg(body.svg);
      setLog(`generated output_id=${body.output_id}`);
      return;
    }
    setLog(JSON.stringify(result.body));
  }

  return (
    <main className="page">
      <section className="panel">
        <h1>Accessibility Handwriting MVP</h1>
        <p>AI生成は常に可視透かし付き、監査ログ保存、目的はアクセシビリティ用途限定。</p>
      </section>

      <section className="panel">
        <h2>Auth</h2>
        <form className="grid" onSubmit={onSignup}>
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="email" />
          <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="password" type="password" />
          <button type="submit">Sign up</button>
          <button type="button" onClick={onLogin}>
            Login
          </button>
        </form>
      </section>

      <section className="panel">
        <h2>Pipeline</h2>
        <div className="grid">
          <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
          <button onClick={onPreprocess} disabled={!authReady}>
            Preprocess
          </button>
          <button onClick={onTrain} disabled={!authReady}>
            Train LoRA
          </button>
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="生成テキスト" />
          <button onClick={onGenerate} disabled={!authReady || styleId === null}>
            Generate
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Output</h2>
        <pre>{log}</pre>
        {svg ? <div className="svg" dangerouslySetInnerHTML={{ __html: svg }} /> : null}
      </section>
    </main>
  );
}

