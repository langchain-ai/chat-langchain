import NextAuth, { NextAuthConfig } from "next-auth";
import Google from "next-auth/providers/google";
import { getDeploymentEnv } from "./deployment-config";

const allowedDomains = process.env.ALLOWED_EMAIL_DOMAINS?.split(',').map(d => d.trim()) || [];
const deploymentEnv = getDeploymentEnv();

// Only configure Google OAuth if we have credentials (internal deployment)
const hasGoogleCredentials = process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET;

export const authConfig: NextAuthConfig = {
  providers: hasGoogleCredentials ? [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      authorization: {
        params: {
          prompt: "consent",
          access_type: "offline",
          response_type: "code",
          // Optionally restrict to your Google Workspace domain
          // hd: "yourcompany.com",
        },
      },
    }),
  ] : [],
  callbacks: {
    async signIn({ user, account, profile }) {
      // Only allow sign-in if email domain is in allowed list
      if (!user.email) {
        return false;
      }

      const emailDomain = user.email.split('@')[1];

      if (allowedDomains.length > 0 && !allowedDomains.includes(emailDomain)) {
        console.log(`Rejected sign-in attempt from domain: ${emailDomain}`);
        return false;
      }

      return true;
    },
    async jwt({ token, user, account }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
      }
      return session;
    },
  },
  // No custom pages - sign-in handled by modal on main page
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60, // 30 days (in seconds)
    // You can customize this:
    // maxAge: 7 * 24 * 60 * 60,  // 7 days
    // maxAge: 60 * 60,            // 1 hour
  },
  // Configure cookies for cross-site context (iframe embedding in Pylon)
  // Only applied for internal deployment
  cookies: deploymentEnv === "internal" ? {
    sessionToken: {
      name: `${deploymentEnv === "internal" ? "__Secure-" : ""}next-auth.session-token`,
      options: {
        httpOnly: true,
        sameSite: "none", // Allow cross-site usage in iframes
        path: "/",
        secure: true, // Required when sameSite is "none"
      },
    },
    callbackUrl: {
      name: `${deploymentEnv === "internal" ? "__Secure-" : ""}next-auth.callback-url`,
      options: {
        httpOnly: true,
        sameSite: "none",
        path: "/",
        secure: true,
      },
    },
    csrfToken: {
      name: `${deploymentEnv === "internal" ? "__Host-" : ""}next-auth.csrf-token`,
      options: {
        httpOnly: true,
        sameSite: "none",
        path: "/",
        secure: true,
      },
    },
  } : undefined,
  // Use a dummy secret for external deployments to prevent errors
  secret: process.env.NEXTAUTH_SECRET || "dummy-secret-not-used-in-external",
};

export const { handlers, auth, signIn, signOut } = NextAuth(authConfig);
