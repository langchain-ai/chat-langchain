// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Strip console calls in production builds
  compiler: {
    removeConsole: process.env.NODE_ENV === "production"
      ? {
          exclude: ["error", "warn"], // Keep errors and warnings for critical issues
        }
      : false,
  },
  // Public app: block iframe embedding by default.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: "frame-ancestors 'self'",
          },
          {
            key: "X-Frame-Options",
            value: "SAMEORIGIN",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
