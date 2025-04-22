import React from "react";
import { GraphProvider } from "./contexts/GraphContext";
import { ChatLangChain } from "./components/ChatLangChain";
import { auth, signIn, signOut } from "@/auth"

export default async function SignIn() {
  const session = await auth();
  console.log(session);
  const user = session?.user
  return user ? (
    <main className="w-full h-full">
      <React.Suspense fallback={null}>
        <GraphProvider>
          <h1> Welcome {user.name} </h1>

          <form
              action={async () => {
              "use server"
              await signOut();
              }}
          >
              <button>Sign Out</button>
          </form>
          <ChatLangChain />
        </GraphProvider>
      </React.Suspense>
    </main>
  ) : 
  (
    <div className="mx-auto my-auto text-center">
      <h1>You are not authenticated</h1>
      <form
        action={async () => {
          "use server"
          await signIn("google")
        }}
      >
        <button type="submit">Sign in with Google</button>
      </form>
    </div>
  )
}

function Page(): React.ReactElement {
  return (
    <main className="w-full h-full">
      <React.Suspense fallback={null}>
        <GraphProvider>
          <ChatLangChain />
        </GraphProvider>
      </React.Suspense>
    </main>
  );
}
