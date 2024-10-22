import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import {NextIntlClientProvider} from 'next-intl';
import {getMessages} from 'next-intl/server';
import {notFound} from 'next/navigation';


const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "StockGPT",
  description: "Chatbot for Stock Trading",
};

export default async function RootLayout({
  children,
  params: {locale}
}: {
  children: React.ReactNode;
  params: {locale: string};
}) {
  const messages = await getMessages();
  return (
    <html lang={locale} className="h-full">
      <body className={`${inter.className} h-full`}>
        <div
          className="flex flex-col h-full w-full"
          style={{ background: "rgb(10, 17, 40)" }}
        >
        <NextIntlClientProvider  messages={messages}>
            {children}
          </NextIntlClientProvider>
        </div>
      </body>
    </html>
  );
}
