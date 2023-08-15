import Image from "next/image";
import Chat from "../app/components/chat";
import { ChatWindow } from "../app/components/ChatWindow";
import { ToastContainer } from 'react-toastify';

export default function Home() {
  const InfoCard = (
    <div className="p-8 rounded bg-[#25252d]">
      <h1 className="text-4xl mb-4">Chat LangChain 🦜🔗</h1>
      <ul>
        <li className="text-l">
          🤝
          <span className="ml-2">
            This template showcases a simple chatbot using{" "}
            <a href="https://js.langchain.com/" target="_blank">
              LangChain.js
            </a>{" "}
            in a{" "}
            <a href="https://nextjs.org/" target="_blank">
              Next.js
            </a>{" "}
            project.
          </span>
        </li>
        <li className="text-l">
          🔁
          <span className="ml-2">
            It also uses the Vercel{" "}
            <a href="https://sdk.vercel.ai/docs" target="_blank">
              AI SDK
            </a>{" "}
            for streaming HTTP responses and the UI.
          </span>
        </li>
        <li className="text-l">
          💻
          <span className="ml-2">
            You can find the prompt and model logic for this use-case in{" "}
            <code>app/api/chat/route.ts</code>.
          </span>
        </li>
        <li>
          🏴‍☠️
          <span className="ml-2">
            By default, the bot is pretending to be a pirate, but you can change
            the prompt to whatever you want!
          </span>
        </li>
        <li className="text-l">
          🎨
          <span className="ml-2">
            The main frontend logic is found in <code>app/page.tsx</code>.
          </span>
        </li>
        <li className="text-l">
          👇<span className="ml-2">Try asking, "What is it like to be a pirate?" or another question below!</span>
        </li>
      </ul>
    </div>
  );
  return (
    <>
      {/* <main className="flex h-[100vh] overflow-hidden flex-col items-center justify-center bg-grey-100">
        <div className="z-10 w-[100vw] w items-center justify-center font-normal text-sm lg:flex lg:max-h-[40rem] sm:max-h-[20rem] text-center p-4 md:p-0"> */}
          {/* <Chat/> */}
          <ToastContainer />
          <ChatWindow
            // endpoint="https://chat-langchain.fly.dev/chat"
            endpoint="http://0.0.0.0:8080/chat"
            emoji="🏴‍☠️"
            titleText="ChatLangChain"
            placeholder="Write your question"
            emptyStateComponent={InfoCard}
          ></ChatWindow>
        {/* </div>
      </main> */}
    </>
  );
}
