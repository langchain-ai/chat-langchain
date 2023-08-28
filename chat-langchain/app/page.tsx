import Image from "next/image";
import { ChatWindow } from "../app/components/ChatWindow";
import { ToastContainer } from "react-toastify";

export default function Home() {
  const InfoCard = (
    <div className="p-8 rounded bg-[#25252d]">
      <h1 className="text-4xl mb-4">Chat LangChain 🦜🔗</h1>
      <ul>
        <li className="text-l">
          🤝
          <span className="ml-2">
            This is a simple chatbot over LangChain&apos;s{" "}
            <a href="https://python.langchain.com/" target="_blank">
              Python Documentation.
            </a>{" "}
          </span>
        </li>
        <li className="text-l">
          🔁
          <span className="ml-2">
            It uses the{" "}
            <a href="https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader">
              RecursiveURLLoader
            </a>{" "}
            in tandem with the{" "}
            <a href="https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.generic.GenericLoader.html">
              GenericLoader
            </a>{" "}
            to ingest all of our API and repository pages.
          </span>
        </li>
        <li className="text-l">
          ⚡️
          <span className="ml-2">
            The documents are chunked and stored in a{" "}
            <a
              href="https://python.langchain.com/docs/integrations/vectorstores/weaviate"
              target="_blank"
            >
              Weaviate
            </a>{" "}
            instance for lightning fast retrieval.
          </span>
        </li>
        <li className="text-l">
          🦾
          <span className="ml-2">
            You can find the prompt and model logic which uses expression language
            and chat history in <code> main.py</code>.
          </span>
        </li>
        <li className="text-l">
          👣
          <span className="ml-2">
            Feedback and Tracing are additional neat features, thanks to {" "}
            <a href="https://smith.langchain.com" target="_blank">LangSmith</a>.
          </span>
        </li>
        <li className="text-l">
          🎨
          <span className="ml-2">
            The main frontend logic is found in <code>app/page.tsx</code>.
          </span>
        </li>
        <li className="text-l">
          👇
          <span className="ml-2">
            Begin by asking, &quot;Show me how to use a RecursiveUrlLoader?&quot; or another
            question below!
          </span>
        </li>
      </ul>
    </div>
  );
  return (
    <>
      <ToastContainer />
      <ChatWindow
        endpoint="https://chat-langchain.fly.dev/chat"
        // emoji="🏴‍☠️"
        titleText="ChatLangChain"
        placeholder="Write your question"
        emptyStateComponent={InfoCard}
      ></ChatWindow>
    </>
  );
}
