import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { Hotjar_GoogleAnalytics_Snippet } from "./hotjar_googleanalytics";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VERA Files SEEK Chatbot | Alpha version",
  description: "Search Experience Elevated to Knockout Disinformation: A VERA Files Fact Checking assistant AI chatbot",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <Hotjar_GoogleAnalytics_Snippet />
      <body className={`${inter.className} h-full`}>
        <div className="flex flex-col h-full w-full text-black bg-[#F9F9F9]">
          <NuqsAdapter>{children}</NuqsAdapter>
        </div>
      </body>
    </html>
  );
}
