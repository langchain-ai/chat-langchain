"use client";

import { signIn } from "next-auth/react";
import { Button } from "@/components/ui/button";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import Image from "next/image";
import { cn } from "@/lib/utils";

interface SignInModalProps {
  open: boolean;
}

export function SignInModal({ open }: SignInModalProps) {
  return (
    <DialogPrimitive.Root open={open} modal>
      <DialogPrimitive.Portal>
        {/* Softer overlay */}
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/20 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />

        {/* Modal content */}
        <DialogPrimitive.Content
          className={cn(
            "fixed left-[50%] top-[50%] z-50 w-full max-w-sm translate-x-[-50%] translate-y-[-50%]",
            "bg-[#0a0a0a]/95 backdrop-blur-md border-2 border-border/50",
            "shadow-[0_32px_64px_-12px_rgba(0,0,0,0.4)]",
            "ring-[3px] ring-primary/20",
            "rounded-3xl p-8",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[state=closed]:slide-out-to-top-[2%] data-[state=open]:slide-in-from-top-[2%]"
          )}
        >
        <DialogPrimitive.Title className="sr-only">
          Sign in to Chat LangChain
        </DialogPrimitive.Title>

        {/* Logo/Brand */}
        <div className="flex flex-col items-center mb-8">
          <Image
            src="/assets/images/Assistant Icon.png"
            alt="Assistant Logo"
            width={88}
            height={88}
            className="object-contain mb-5"
          />
          <h1 className="text-2xl font-bold text-foreground mb-2">Welcome to</h1>
          <h2 className="text-2xl font-bold text-foreground mb-3">
            Chat <span className="text-primary">LangChain</span>
          </h2>
          <p className="text-sm text-muted-foreground text-center">
            Sign in with your work account to continue
          </p>
        </div>

        <div className="space-y-6">
          <Button
            onClick={() => signIn("google", { redirectTo: "/" })}
            className="w-full h-12 bg-white hover:bg-gray-50 text-gray-900 border-2 border-gray-200 hover:border-gray-300 shadow-md hover:shadow-xl transition-all duration-300 font-medium rounded-xl hover:scale-[1.02] active:scale-[0.98]"
          >
            <svg
              className="mr-2 h-5 w-5"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
            </svg>
            Continue with Google
          </Button>

          <div className="text-center space-y-2">
            <p className="text-[11px] text-muted-foreground/80 leading-relaxed">
              By signing in, you agree to our Terms of Service and Privacy Policy.
            </p>
            <p className="text-[11px] text-muted-foreground/60">
              Need help? Contact your administrator
            </p>
          </div>
        </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
