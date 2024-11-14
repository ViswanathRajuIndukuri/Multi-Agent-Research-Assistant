import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone using the Pinecone class
pc = Pinecone(
    api_key="90d8c180-9807-4be2-8628-ead31d7f8902"  # replace with your actual API key
)

# Create or connect to an index
index_name = "gpt-4o-research-agent"  # replace with your actual index name
dimension = 1536  # replace with the dimension size of your embeddings
metric = "cosine"  # or "euclidean", depending on your needs

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # replace with your actual region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Retrieve and display index details
index_info = pc.describe_index(index_name)
print(index_info)
