import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from openai import OpenAI


# Load environment variables
load_dotenv()

# Set up Together API client using OpenAI compatible interface
api_key = os.getenv("TOGETHER_API_KEY")
base_url = os.getenv("TOGETHER_BASE_URL")

# Path to saved ChromaDB
CHROMA_PATH = "chroma_db"

# Prompt Template
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """Give a detailed answer to the question based on the context below:

    Question: {question}

    Context: {context}
    """
)

def query_rag(query_text):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    # Debugging: Check the total number of documents in vector store
    # print(f"Total documents in vector store: {vector_store._collection.count()}")

    # Retrieve the most relevant results for the query
    results = vector_store.similarity_search_with_relevance_scores(query_text, k=7)

    # Debugging: Check the retrieved results and their relevance scores
    print(f"Retrieved results: {results}")
    if not results:
        print("No matching results found.")
        return
    
    if results[0][1] < 0.7:
        print(f"Low confidence match found with score: {results[0][1]}")
        print("\n")

    # Build context from retrieved results
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])
    print(f"Context text: {context_text}")

    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    # print(f"Formatted prompt: {prompt}")

    # Make the API call to Mistral
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"role": "user", "content": str(prompt)}]
    )

    # Debugging: Print the raw response from Mistral
    # print(f"Raw response: {response}")

    # Extract and print model response
    response_text = response.choices[0].message.content
    # print(f"Model response: {response_text}")

    # Extract sources (metadata) for the context
    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    # Return the final formatted response
    return formatted_response, response_text

if __name__ == "__main__":
    user_question = input("Ask your question: \n")
    result, _ = query_rag(user_question)
    print(result)
