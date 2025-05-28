import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ThemeProvider } from "next-themes";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NatureAlpha Intelligence",
  description: "Intelligent Chat by NatureAlpha",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Runtime URL patching to ensure correct API URL
              (function() {
                try {
                  // Override for when the client is created
                  var originalConsoleLog = console.log;
                  console.log = function() {
                    // Check if this is the API URL log
                    if (arguments.length > 0 && 
                        typeof arguments[0] === 'string' && 
                        arguments[0] === 'Using API URL:' && 
                        arguments[1] && 
                        arguments[1].includes('localhost:2024')) {
                      
                      // Redirect to the correct URL
                      const correctUrl = "${process.env.NEXT_PUBLIC_API_URL || process.env.API_BASE_URL || 'https://naturealpha-docs-6f63c1e32335558f86984cf800d1f815.us.langgraph.app'}";
                      
                      // Silently override without logging sensitive keys
                      document.dispatchEvent(new CustomEvent('override-api-url', { 
                        detail: { url: correctUrl } 
                      }));
                      
                      return;
                    }
                    
                    // Call the original console.log for other messages
                    originalConsoleLog.apply(console, arguments);
                  };
                } catch (e) {
                  // Silent error handling
                }
              })();
            `,
          }}
        />
      </head>
      <body className={`${inter.className} h-full`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div
            className="flex flex-col h-full w-full"
            style={{ background: "rgb(38, 38, 41)" }}
          >
            <NuqsAdapter>{children}</NuqsAdapter>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
