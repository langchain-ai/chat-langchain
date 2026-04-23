import type { Metadata } from "next";
import { Inter, Inconsolata } from "next/font/google";
import { NuqsAdapter } from 'nuqs/adapters/next/app'
import Script from "next/script";
import "./globals.css";
import { SegmentProvider } from "@/components/providers/segment-provider";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const inconsolata = Inconsolata({
  variable: "--font-inconsolata",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Chat LangChain",
  description: "AI-powered guide to the LangChain ecosystem",
  icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    shortcut: '/favicon-32x32.png',
    apple: '/favicon-32x32.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const segmentWriteKey = process.env.NEXT_PUBLIC_SEGMENT_WRITE_KEY;

  return (
    <html lang="en">
      <head>
        {/* Segment Analytics */}
        {segmentWriteKey && (
          <Script
            id="segment-script"
            strategy="beforeInteractive"
            dangerouslySetInnerHTML={{
              __html: `
!function(){var i="analytics",analytics=window[i]=window[i]||[];if(!analytics.initialize)if(analytics.invoked)window.console&&console.error&&console.error("Segment snippet included twice.");else{analytics.invoked=!0;analytics.methods=["trackSubmit","trackClick","trackLink","trackForm","pageview","identify","reset","group","track","ready","alias","debug","page","screen","once","off","on","addSourceMiddleware","addIntegrationMiddleware","setAnonymousId","addDestinationMiddleware","register"];analytics.factory=function(e){return function(){if(window[i].initialized)return window[i][e].apply(window[i],arguments);var n=Array.prototype.slice.call(arguments);if(["track","screen","alias","group","page","identify"].indexOf(e)>-1){var c=document.querySelector("link[rel='canonical']");n.push({__t:"bpc",c:c&&c.getAttribute("href")||void 0,p:location.pathname,u:location.href,s:location.search,t:document.title,r:document.referrer})}n.unshift(e);analytics.push(n);return analytics}};for(var n=0;n<analytics.methods.length;n++){var key=analytics.methods[n];analytics[key]=analytics.factory(key)}analytics.load=function(key,n){var t=document.createElement("script");t.type="text/javascript";t.async=!0;t.setAttribute("data-global-segment-analytics-key",i);t.src="https://cdn.segment.com/analytics.js/v1/" + key + "/analytics.min.js";var r=document.getElementsByTagName("script")[0];r.parentNode.insertBefore(t,r);analytics._loadOptions=n};analytics._writeKey="${segmentWriteKey}";;analytics.SNIPPET_VERSION="5.2.0";
analytics.load("${segmentWriteKey}");
analytics.page();
}}();
              `,
            }}
          />
        )}
      </head>
      <body
        className={`${inter.variable} ${inconsolata.variable} antialiased`}
        suppressHydrationWarning
      >
        <SegmentProvider>
          <NuqsAdapter>
            {children}
          </NuqsAdapter>
        </SegmentProvider>
      </body>
    </html>
  );
}
