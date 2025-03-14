import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Force consistent output regardless of environment to avoid hydration mismatches
  output: "export",
  
  // Configure images to work with export
  images: {
    unoptimized: true,
  },
  
  // Optimize for production builds and consistent CSS handling
  productionBrowserSourceMaps: false,
  
  // Make the output consistently formatted
  experimental: {
    // Disable optimizeCss since it requires critters
    optimizeCss: false,
  },
  
  // External packages that should be resolved server-side
  serverExternalPackages: [],
  
  // Disable TypeScript checking during build
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
  
  // Disable ESLint during build
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  
  // Enhance stability during development
  webpack(config) {
    return config;
  },
};

export default nextConfig;
