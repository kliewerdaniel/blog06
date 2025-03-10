import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Force consistent output regardless of environment to avoid hydration mismatches
  output: "standalone",
  
  // Optimize for production builds and consistent CSS handling
  productionBrowserSourceMaps: false,
  
  // Make the output consistently formatted
  experimental: {
    // Use inline CSS to avoid class name discrepancies 
    optimizeCss: true,
  },
  
  // External packages that should be resolved server-side
  serverExternalPackages: [],
  
  // Enhance stability during development
  webpack(config) {
    return config;
  },
};

export default nextConfig;

module.exports = {
  output: 'export',
};