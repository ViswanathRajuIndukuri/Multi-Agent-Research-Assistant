import getpass
import os
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import time

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Ensure API keys are read from .env
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API keys are loaded
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found in environment variables.")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

pc = Pinecone(api_key=pinecone_api_key)

index_name = "pdf-investment-model-validation-a-guide-for"  # change if desired
#index_name = "gpt-4o-research-agent" 

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# 1. Create a retriever from your vector store
retriever = vector_store.as_retriever()

# 2. Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# 4. Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 5. Create a function to handle queries
def get_answer(question: str):
    response = qa_chain.invoke({
        "query": question
    })
    # Extract answer and sources
    answer = response["result"]
    return {
        "answer": answer,
    }

# 6. Example usage
question = "investment-model-validation"
result = get_answer(question)
print("\nAnswer:", result)