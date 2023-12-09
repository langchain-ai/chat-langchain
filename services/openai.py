from typing import List
import openai
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # Call the OpenAI API to get the embeddings using azure
    openai.api_type = "azure"
    openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = "text-embedding-ada-002"

    try:
        if deployment is None:
            print("Creating embeddings using OpenAI")
            response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        else:
            print("Creating embeddings using Azure")
            response = openai.Embedding.create(input=texts, deployment_id=deployment)

        print("Response received:", response)
        
        # Extract the embedding data from the response
        data = response["data"]  # type: ignore

        # Return the embeddings as a list of lists of floats
        return [result["embedding"] for result in data]
    except openai.error.OpenAIError as e:
        # Log more detailed error information
        print(f"An error occurred: {e}")
        raise
    except Exception as e:
        # Catch any other exceptions that might occur
        print(f"An unexpected error occurred: {e}")
        raise
