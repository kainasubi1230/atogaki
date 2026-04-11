import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Accessibility Handwriting MVP",
  description: "AI-assisted accessibility handwriting with visible watermark and audit logs."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}

