import datetime
import os

import gradio as gr
import langchain
import weaviate
from chain import get_new_chain1
from langchain.vectorstores import Weaviate

WEAVIATE_URL = os.environ["WEAVIATE_URL"]


def get_weaviate_store():
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    return Weaviate(client, "Paragraph", "content", attributes=["source"])


def set_openai_api_key(api_key, agent):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        vectorstore = get_weaviate_store()
        qa_chain = get_new_chain1(vectorstore)
        os.environ["OPENAI_API_KEY"] = ""
        return qa_chain


def chat(inp, history, agent):
    history = history or []
    if agent is None:
        history.append((inp, "Please paste your OpenAI key to use"))
        return history, history
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    history = history or []
    output = agent({"question": inp, "chat_history": history})
    answer = output["answer"]
    history.append((inp, answer))
    print(history)
    return history, history


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain AI</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What are agents?",
            "How do I summarize a long document?",
            "What types of memory exist?",
        ],
        inputs=message,
    )

    gr.HTML(
        """
    This simple application is an implementation of ChatGPT but over an external dataset (in this case, the LangChain documentation)."""
    )

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox, agent_state],
        outputs=[agent_state],
    )

block.launch(debug=True)
