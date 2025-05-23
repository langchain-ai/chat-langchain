import streamlit as st
from langchain_core.runnables.remote import RemoteRunnable
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
DEFAULT_LANGSERVE_URL = "http://localhost:8000/chat_langchain/" # Replace 'chat_langchain' if your LangServe endpoint is different
# Fetch model keys from backend.graph.py (or define them statically if easier for now)
# These should match the keys in llm.configurable_alternatives in graph.py
AVAILABLE_MODELS = {
    "OpenAI GPT-3.5": "openai_gpt_3_5_turbo",
    "Ollama (Default)": "ollama_chat", # Assuming OLLAMA_MODEL_KEY = "ollama_chat"
    "Anthropic Claude 3 Haiku": "anthropic_claude_3_haiku",
    "Fireworks Mixtral": "fireworks_mixtral",
    "Google Gemini Pro": "google_gemini_pro",
    "Cohere Command": "cohere_command",
    "Groq Llama 3": "groq_llama_3",
}

st.set_page_config(page_title="Chat LangChain Interface", page_icon="🦜")
st.title("Chat LangChain Interface 🦜")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    langserve_url = st.text_input("LangServe URL", value=DEFAULT_LANGSERVE_URL)
    
    model_display_name = st.selectbox(
        "Choose LLM Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0 # Default to the first model in the list
    )
    selected_model_key = AVAILABLE_MODELS[model_display_name]

# --- Initialize Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # The AIMessage from the backend graph should have the answer in `content`
            # and sources in `documents` (or however the graph structures it)
            st.markdown(message.content) 
            # Display sources if available (this part might need adjustment based on backend output)
            if message.additional_kwargs.get("documents"):
                 with st.expander("Sources"):
                    for doc in message.additional_kwargs["documents"]:
                        st.markdown(f"- **{doc.metadata.get('title', 'Unknown Title')}**: [{doc.metadata.get('source', 'No URL')}]({doc.metadata.get('source', '')})")
                        # st.markdown(doc.page_content) # Optionally show snippets

# --- Handle User Input ---
user_query = st.chat_input("Ask a question about LangChain, Milvus, or Ollama...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        try:
            # Ensure the RemoteRunnable is configured with the selected model
            chain = RemoteRunnable(langserve_url, config={"configurable": {"model_name": selected_model_key}})
            
            # Prepare input for the chain based on backend.graph.AgentState
            # The graph expects {'messages': [HumanMessage(content=user_query)]} for the first turn
            # or a list of messages for subsequent turns.
            
            # For simplicity, let's send the last message. The graph handles history.
            # The graph's input schema is {"messages": [BaseMessage]}
            # The output schema is {"messages": [AIMessage], "answer": str, "documents": list[Document]}

            # The input to the graph is the full list of messages.
            # The graph itself handles condensing the question with history.
            # input_messages = st.session_state.chat_history # Send the whole history
            
            # The current chat-langchain graph expects the 'query' directly
            # and 'messages' for history. Let's adapt to send the last query
            # and the history separately if needed by the specific graph state.
            # However, the AgentState in graph.py uses `add_messages` for the 'messages' field,
            # implying it expects a list of messages.
            
            # The graph's input schema is `AgentState` which includes `messages`.
            # The `add_messages` updater on `AgentState.messages` means we should pass the new message.
            # The graph itself will manage the history.

            # The simplest input to the graph is a dictionary with a "messages" key
            # containing a list of HumanMessage or AIMessage objects.
            # The graph input type is: {'messages': [('human', "your question")]}
            # or more generally List[Tuple[str, str]] or List[BaseMessage]
            
            # Let's construct the input as a list of previous messages + current query
            # The graph's `add_messages` will append the latest one.
            # The runnable's `invoke` method will take the input that matches the graph's input schema.
            # The graph's input schema is AgentState, but when served via LangServe,
            # it typically exposes specific input fields.
            # The default LangServe endpoint for a graph takes the fields of the AgentState.
            # So we need to provide `{"messages": [HumanMessage(content=user_query)]}`
            # If history is managed by the graph, we pass the history.

            # The `chat_langchain` graph's entry point routes based on len(state.messages).
            # For a new query, it's one message.
            
            # Let's send the current user_query as the content of the last HumanMessage.
            # The graph's `add_messages` will handle appending it.
            # The graph will handle history internally.
            
            # The input to the runnable should match the input schema of the *exposed* LangServe endpoint.
            # Typically, for a stateful graph, you pass the current input, and the server manages state
            # or you pass the whole state.
            # The `chat_langchain` graph is stateful.
            # For a stateful graph served with `add_routes(graph, path="/chat")`
            # the input is usually the input to the first node or an update to the state.

            # The `chat_langchain` graph is compiled and likely served.
            # The input to its `/invoke` endpoint would be the `AgentState` fields.
            # We need to provide `{"messages": [HumanMessage(content=user_query)]}`.
            # The graph itself handles the history via the `AgentState`.
            
            # Let's try sending only the new message.
            # If the backend is properly stateful per conversation (e.g. using configurable session_id),
            # it will manage history.
            # If not, we might need to send the whole history st.session_state.chat_history

            # The current chat_langchain graph is not explicitly managing sessions in the Python code shown.
            # It re-evaluates history on each call. So, we MUST send the history.
            
            payload = {"messages": [msg.dict() for msg in st.session_state.chat_history]}


            # Stream the response
            for chunk in chain.stream(payload):
                # chunk is an AgentState partial dictionary
                # We are interested in the "messages" field which contains AIMessage chunks
                if "messages" in chunk:
                    ai_message_chunk = chunk["messages"][-1] # Get the last message (the AI's response)
                    if isinstance(ai_message_chunk, AIMessage):
                        full_response_content += ai_message_chunk.content
                        message_placeholder.markdown(full_response_content + "▌")
                    elif isinstance(ai_message_chunk, dict) and "content" in ai_message_chunk: # Sometimes it's a dict
                        full_response_content += ai_message_chunk["content"]
                        message_placeholder.markdown(full_response_content + "▌")


            # Once streaming is done, get the final AI message with all details
            # The final output of the graph is the full AgentState.
            final_output_state = chain.invoke(payload)
            ai_response_message = final_output_state["messages"][-1] # Should be the complete AIMessage

            message_placeholder.markdown(ai_response_message.content)
            st.session_state.chat_history.append(ai_response_message)

            # Display sources from the final response message
            if ai_response_message.additional_kwargs.get("documents"): # graph.py puts docs in AgentState.documents
                 with st.expander("Sources"):
                    for doc in ai_response_message.additional_kwargs["documents"]: # This might be final_output_state["documents"]
                        doc_md = doc.get("metadata", {}) if isinstance(doc, dict) else doc.metadata
                        page_content = doc.get("page_content", "") if isinstance(doc, dict) else doc.page_content
                        st.markdown(f"- **{doc_md.get('title', 'Unknown Title')}**: [{doc_md.get('source', 'No URL')}]({doc_md.get('source', '')})")
                        # st.markdown(page_content) # Optionally show snippets
            elif final_output_state.get("documents"): # If it's directly in the state
                 with st.expander("Sources"):
                    for doc_data in final_output_state["documents"]:
                        doc_md = doc_data.get("metadata", {}) if isinstance(doc_data, dict) else doc_data.metadata
                        page_content = doc_data.get("page_content", "") if isinstance(doc_data, dict) else doc_data.page_content
                        st.markdown(f"- **{doc_md.get('title', 'Unknown Title')}**: [{doc_md.get('source', 'No URL')}]({doc_md.get('source', '')})")

        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
            # Add the error as a message to history to acknowledge
            error_message = AIMessage(content=f"Sorry, I encountered an error: {e}")
            st.session_state.chat_history.append(error_message)
            message_placeholder.markdown(error_message.content)

# Add a note about running the backend
st.sidebar.info("Ensure the LangServe Python backend is running for this interface to work.")
