"use client";

import { useEffect, useState } from "react";
import { postJSON } from "../lib/api";
import { ScreenState } from "../lib/types";
import { SplashScreen } from "../components/SplashScreen";
import { ScanScreen } from "../components/ScanScreen";
import { CanvasScreen } from "../components/CanvasScreen";

export default function Page() {
  const [screen, setScreen] = useState<ScreenState>("splash");
  const [token, setToken] = useState("");
  const [userId, setUserId] = useState<number | null>(null);
  const [styleId, setStyleId] = useState<number | null>(null);

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
    };

    initApp();

    // 2. Wait a minimum of 2.5 seconds to show the beautiful splash, unless skipped
    const timer = setTimeout(() => {
      setScreen((prev) => (prev === "splash" ? "scan" : prev));
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="app-container">
      {screen === "splash" && (
        <SplashScreen onNext={() => setScreen("scan")} />
      )}

      {screen === "scan" && (
        <ScanScreen 
          token={token} 
          userId={userId} 
          onComplete={(newStyleId) => {
            if (newStyleId) setStyleId(newStyleId);
            setScreen("canvas");
          }} 
        />
      )}

      {screen === "canvas" && (
        <CanvasScreen token={token} userId={userId} styleId={styleId} />
      )}
    </div>
  );
}
