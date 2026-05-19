"use client";

import { useCallback, useEffect, useState } from "react";
import { postJSON } from "../lib/api";
import { ScreenState } from "../lib/types";
import { SplashScreen } from "../components/SplashScreen";
import { ScanScreen } from "../components/ScanScreen";
import { CanvasScreen } from "../components/CanvasScreen";

const AUTH_CACHE_KEY = "machine_guest_auth_v1";
const AUTH_CACHE_TTL_MS = 55 * 60 * 1000;

type CachedAuth = {
  token: string;
  userId: number;
  expiresAt: number;
};

export default function Page() {
  const [screen, setScreen] = useState<ScreenState>("splash");
  const [token, setToken] = useState("");
  const [userId, setUserId] = useState<number | null>(null);
  const [styleId, setStyleId] = useState<number | null>(null);
  const [authStatus, setAuthStatus] = useState<"loading" | "ready" | "error">("loading");

  const loadCachedAuth = (): CachedAuth | null => {
    try {
      const raw = localStorage.getItem(AUTH_CACHE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as CachedAuth;
      if (
        typeof parsed.token !== "string" ||
        typeof parsed.userId !== "number" ||
        typeof parsed.expiresAt !== "number"
      ) {
        return null;
      }
      if (parsed.expiresAt <= Date.now()) {
        localStorage.removeItem(AUTH_CACHE_KEY);
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  };

  const saveCachedAuth = (nextToken: string, nextUserId: number) => {
    const payload: CachedAuth = {
      token: nextToken,
      userId: nextUserId,
      expiresAt: Date.now() + AUTH_CACHE_TTL_MS,
    };
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(payload));
  };

  const initAuth = useCallback(async () => {
    setAuthStatus("loading");
    const cached = loadCachedAuth();
    if (cached) {
      setToken(cached.token);
      setUserId(cached.userId);
      setAuthStatus("ready");
      return;
    }

    const email = `guest_${Date.now()}@example.com`;
    const pwd = "password";
    try {
      const res = await postJSON("/auth/signup", { email, password: pwd });
      if (res.status === 200) {
        const body = res.body as { access_token: string; user_id: number };
        setToken(body.access_token);
        setUserId(body.user_id);
        saveCachedAuth(body.access_token, body.user_id);
        setAuthStatus("ready");
      } else {
        setAuthStatus("error");
      }
    } catch (e) {
      console.error("Auto signup failed", e);
      setAuthStatus("error");
    }
  }, []);

  useEffect(() => {
    initAuth();

    const timer = setTimeout(() => {
      setScreen((prev) => (prev === "splash" ? "scan" : prev));
    }, 400);

    return () => clearTimeout(timer);
  }, [initAuth]);

  useEffect(() => {
    if (authStatus !== "error") return;
    const retryTimer = setTimeout(() => {
      initAuth();
    }, 3000);
    return () => clearTimeout(retryTimer);
  }, [authStatus, initAuth]);

  return (
    <div className="app-container">
      {screen === "splash" && <SplashScreen onNext={() => setScreen("scan")} />}

      {screen === "scan" && (
        <ScanScreen
          token={token}
          userId={userId}
          authStatus={authStatus}
          onRetryAuth={initAuth}
          onComplete={(newStyleId) => {
            if (newStyleId) setStyleId(newStyleId);
            setScreen("canvas");
          }}
        />
      )}

      {screen === "canvas" && <CanvasScreen token={token} userId={userId} styleId={styleId} />}
    </div>
  );
}
