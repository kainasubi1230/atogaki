"use client";

import { useCallback, useEffect, useState } from "react";
import { postJSON } from "../lib/api";
import { ScreenState } from "../lib/types";
import { SplashScreen } from "../components/SplashScreen";
import { ScanScreen } from "../components/ScanScreen";
import { CanvasScreen } from "../components/CanvasScreen";

const AUTH_CACHE_KEY = "machine_guest_auth_v1";
const GUEST_IDENTITY_KEY = "machine_guest_identity_v1";
const AUTH_CACHE_TTL_MS = 55 * 60 * 1000;

type CachedAuth = {
  token: string;
  userId: number;
  expiresAt: number;
};

type GuestIdentity = {
  email: string;
  password: string;
};

function randomGuestSuffix(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
  }
  return `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export default function Page() {
  const [screen, setScreen] = useState<ScreenState>("splash");
  const [token, setToken] = useState("");
  const [userId, setUserId] = useState<number | null>(null);
  const [styleId, setStyleId] = useState<number | null>(null);
  const [authStatus, setAuthStatus] = useState<"loading" | "ready" | "error">("loading");
  const [authError, setAuthError] = useState("");

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

  const loadOrCreateGuestIdentity = (): GuestIdentity => {
    try {
      const raw = localStorage.getItem(GUEST_IDENTITY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as GuestIdentity;
        if (
          typeof parsed.email === "string" &&
          parsed.email.length > 0 &&
          typeof parsed.password === "string" &&
          parsed.password.length >= 8
        ) {
          return parsed;
        }
      }
    } catch {
      // ignore cache parse errors
    }
    const created: GuestIdentity = {
      email: `guest_${randomGuestSuffix()}@example.com`,
      password: "password123",
    };
    localStorage.setItem(GUEST_IDENTITY_KEY, JSON.stringify(created));
    return created;
  };

  const rotateGuestIdentity = (): GuestIdentity => {
    const created: GuestIdentity = {
      email: `guest_${randomGuestSuffix()}@example.com`,
      password: "password123",
    };
    localStorage.setItem(GUEST_IDENTITY_KEY, JSON.stringify(created));
    return created;
  };

  const resolveErrorDetail = (body: unknown): string => {
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
    return "auth_init_failed";
  };

  const initAuth = useCallback(async () => {
    setAuthStatus("loading");
    setAuthError("");
    const cached = loadCachedAuth();
    if (cached) {
      setToken(cached.token);
      setUserId(cached.userId);
      setAuthStatus("ready");
      return;
    }

    const tryAuthWithIdentity = async (identity: GuestIdentity): Promise<{ token: string; userId: number } | null> => {
      const loginRes = await postJSON("/auth/login", {
        email: identity.email,
        password: identity.password,
      });
      if (loginRes.status === 200) {
        const body = loginRes.body as { access_token: string; user_id: number };
        return { token: body.access_token, userId: body.user_id };
      }

      const signupRes = await postJSON("/auth/signup", {
        email: identity.email,
        password: identity.password,
      });
      if (signupRes.status === 200) {
        const body = signupRes.body as { access_token: string; user_id: number };
        return { token: body.access_token, userId: body.user_id };
      }

      if (signupRes.status === 409) {
        const retryLogin = await postJSON("/auth/login", {
          email: identity.email,
          password: identity.password,
        });
        if (retryLogin.status === 200) {
          const body = retryLogin.body as { access_token: string; user_id: number };
          return { token: body.access_token, userId: body.user_id };
        }
      }
      setAuthError(resolveErrorDetail(signupRes.body));
      return null;
    };

    const identity = loadOrCreateGuestIdentity();
    try {
      const first = await tryAuthWithIdentity(identity);
      if (first) {
        setToken(first.token);
        setUserId(first.userId);
        saveCachedAuth(first.token, first.userId);
        setAuthStatus("ready");
        return;
      }

      // Fallback: rotate guest identity once in case persisted identity became inconsistent.
      localStorage.removeItem(AUTH_CACHE_KEY);
      const rotated = rotateGuestIdentity();
      const second = await tryAuthWithIdentity(rotated);
      if (second) {
        setToken(second.token);
        setUserId(second.userId);
        saveCachedAuth(second.token, second.userId);
        setAuthStatus("ready");
        return;
      }
      setAuthStatus("error");
    } catch (e) {
      console.error("Auto signup failed", e);
      setAuthError(e instanceof Error ? e.message : "auth_init_exception");
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
          authError={authError}
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
