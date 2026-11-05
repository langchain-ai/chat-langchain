import { notFound } from "next/navigation";
import { MacawChatDemo } from "@/components/dev/macaw-foundations-smoke";

export const dynamic = "force-dynamic";
export const metadata = { robots: { index: false, follow: false } };

export default function MacawSmokePage() {
  // Opt in locally when checking the production build; this route otherwise 404s.
  if (process.env.MACAW_SMOKE_TEST !== "1") notFound();

  return <MacawChatDemo />;
}
