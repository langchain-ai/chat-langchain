import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Script from 'next/script'; // Import the Script component for Google Analytics

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CropTalk virtual assistant",
  description: "Crop insurance assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
            {/* Google Analytics */}
      <Script
        strategy="afterInteractive"
        src="https://www.googletagmanager.com/gtag/js?id=G-DRFPTM9QR0"
      />
      <Script
        id="google-analytics"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-DRFPTM9QR0', {
              page_path: window.location.pathname,
            });
          `,
        }}
      />
      {/* End of Google Analytics */}
      <body className={`${inter.className} h-full`}>
        <div
          className="flex flex-col h-full md:p-8"
          style={{ background: "rgb(38, 38, 41)" }}
        >
          {children}
        </div>
      </body>
    </html>
  );
}
