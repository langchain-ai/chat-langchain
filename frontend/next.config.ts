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
  // Configure headers for iframe embedding (internal deployment only)
  async headers() {
    const deploymentEnv = process.env.NEXT_PUBLIC_DEPLOYMENT_ENV;

    // Only apply these headers for internal deployment
    if (deploymentEnv === "internal") {
      return [
        {
          source: "/:path*",
          headers: [
            {
              key: "Content-Security-Policy",
              value: "frame-ancestors 'self' https://app.usepylon.com",
            },
            // X-Frame-Options is deprecated but included for older browsers
            {
              key: "X-Frame-Options",
              value: "ALLOW-FROM https://app.usepylon.com",
            },
          ],
        },
      ];
    }

    // For external deployment, block all iframe embedding
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
