import React from "react";
import { GraphProvider } from "./contexts/GraphContext";
import { ChatLangChain } from "./components/ChatLangChain";
import { Card } from "./components/ui/card";
import NextImage from "next/image";
import { auth, signIn, signOut } from "@/auth";

export default async function SignIn() {
  const session = await auth();
  console.log(session);
  const user = session?.user;
  return user ? (
    <main className="w-full h-full">
      <React.Suspense fallback={null}>
        <GraphProvider>
          <ChatLangChain test={user.name} />
        </GraphProvider>
      </React.Suspense>
    </main>
  ) : 
  (
    <main className="flex mx-auto my-auto text-center justify-center items-center">
      <Card className="flex flex-col justify-center items-center w-[90%] lg:w-[60%] p-16">
        <NextImage
          src="/images/verafiles_banner.png"
          className=""
          alt="Verafiles Logo"
          width={208}
          height={208}
        />
        <h1 className="mt-4 font-bold text-3xl">Welcome to SEEK!</h1>
        <h2 className="mt-2 font-semibold text-lg">Check your facts in less than a minute.</h2>
        <p className="mt-6 font-light text-sm">By clicking “Login”, you agree to our Terms. Learn how we process your data in our Privacy Policy and Cookie Policy.</p>
        <Card className="mt-6 w-[50%] cursor-pointer">
          <form className="flex flex-row justify-center"
            action={async () => {
              "use server"
              await signIn("google")
            }}
          >
            <NextImage
              src="/images/Google Logo.svg"
              className=""
              alt="Google Logo"
              width={20}
              height={20}
            />
            <button className="p-2 text-gray-600 text-xs sm:text-sm md:text-base" type="submit">Log In with Google</button>
          </form>
        </Card>
      </Card>
    </main>
  )
}

// function Page(): React.ReactElement {
//   return (
//     <main className="w-full h-full">
//       <React.Suspense fallback={null}>
//         <GraphProvider>
//           <ChatLangChain />
//         </GraphProvider>
//       </React.Suspense>
//     </main>
//   );
// }
