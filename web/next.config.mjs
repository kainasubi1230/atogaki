/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Separate dev/prod build outputs to avoid chunk races like missing ./65.js.
  distDir:
    process.env.NEXT_DIST_DIR ||
    (process.env.NODE_ENV === "development" ? ".next-dev" : ".next"),
};

export default nextConfig;

