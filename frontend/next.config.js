/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    API_BASE_URL: process.env.API_BASE_URL,
    LANGCHAIN_API_KEY: process.env.LANGCHAIN_API_KEY
  },
  // Add webpack config to include environment variables
  webpack: (config, { isServer }) => {
    // This is important for making environment variables accessible in client components
    config.plugins.forEach((plugin) => {
      if (plugin.definitions && plugin.definitions['process.env.NEXT_PUBLIC_API_URL']) {
        console.log('Webpack: Found NEXT_PUBLIC_API_URL definition:', 
          plugin.definitions['process.env.NEXT_PUBLIC_API_URL']);
      }
    });
    
    return config;
  },
  // Add this to force proper rendering of client components with environment variables
  reactStrictMode: true,
};

// Print config values for debugging
console.log('Next.js config environment variables:');
console.log('  NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
console.log('  API_BASE_URL:', process.env.API_BASE_URL);
console.log('  LANGCHAIN_API_KEY available:', !!process.env.LANGCHAIN_API_KEY);

module.exports = nextConfig;
