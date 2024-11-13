import os
from pinecone import Pinecone
from dotenv import load_dotenv
# Initialize Pinecone using the Pinecone class
# Configure client

load_dotenv()

# Verify that API keys are loaded
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# List all indexes
index_list = pc.list_indexes().names()  # Fetches the list of index names
print("Available indexes:")
for idx in index_list:
    print(idx)