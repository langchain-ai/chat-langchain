from langsmith import Client
from langchain.prompts import load_prompt

client = Client()

def list_prompts():
    try:
        prompts = client.list_prompts()
        print("\nPrompt IDs to use with client.pull_prompt():")
        print("----------------------------------------")
        for prompt in prompts:
            # Format: org_name/prompt_name
            pull_id = f"{prompt}"
            print(f"client.pull_prompt(\"{pull_id}\")")
    except Exception as e:
        print("Error listing prompts:", str(e))

def test_langsmith_connection():
    try:
        # Example using one of your prompts
        prompt = client.pull_prompt("langchain-ai/chat-langchain-generate-queries-prompt")
        print("Successfully connected to LangSmith!")
        print("Prompt content:", prompt.messages[0].prompt.template)
    except Exception as e:
        print("Error connecting to LangSmith:", str(e))

if __name__ == "__main__":
    #list_prompts()
    test_langsmith_connection()