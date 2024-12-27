# Research Assistant Using LangGraph and Multi-Agent Architecture

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com)
[![LangChain](https://img.shields.io/badge/🦜️_LangChain-008080?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6F61?style=for-the-badge&logo=graph&logoColor=white)](https://github.com/langchain-ai/langgraph)

## Description

This project implements an advanced research assistant system that combines document processing, multi-agent architecture, and vector search capabilities. The system uses Apache Airflow for document processing pipelines, integrates multiple specialized agents (Web Search, Arxiv, and RAG) for comprehensive research, and provides a user-friendly interface through Streamlit. The system leverages LangGraph for agent orchestration and Pinecone for efficient vector storage and retrieval.

**Documentation**: [Codelab documentation Link](https://codelabs-preview.appspot.com/?file_id=1tD-KkdWDk6lJoKLtsfJHdoMYOAPC0y6VgCMH6rwFH1s#0)

**Demo Video Link**: [Demo Video Link](https://drive.google.com/drive/folders/1s_75nS7xyRuR5hFrxht7HibdO7kkWYEO?usp=drive_link)

**Application URL**: http://viswanath.me:8501/

**Backend Service Link**: http://viswanath.me:8000/docs

## Architecture
![research_assistant_system](https://github.com/user-attachments/assets/bf632d64-bd67-4ff2-b93e-455774572cb0)

![langgraph_agents_system](https://github.com/user-attachments/assets/41a3d4be-819a-45bb-b8d4-e70177b744a8)

## About

**Problem**

The challenge is to build an end-to-end research tool that can process documents, coordinate multiple specialized agents, and provide a comprehensive research interface for users. The system needs to handle document processing, vector storage, and multi-agent coordination while maintaining a user-friendly interface.

**Scope**
+ **Document Processing**: Automate the document processing via Docling and vector storage using Pinecone using Airflow
+ **Multi-Agent System**: Implement and coordinate multiple specialized agents using LangGraph.
+ **Vector Search**: Enable efficient document retrieval using Pinecone.
+ **User Interface**: Create an intuitive research interface using Streamlit and FastAPI.

**Outcomes**
+ A fully functional research assistant system with multiple specialized agents
+ Efficient document processing and vector storage pipeline
+ User-friendly interface for conducting research
+ Export capabilities for research findings

## Application Workflow

1. **Document Processing Pipeline (Airflow)**:
   - Process documents using docling (images and tables are stored seperately)
   - generate embeddings using openAI
   - Store vectors in Pinecone
   - Manage document metadata

2. **Multi-Agent System**:
   - Web Search Agent for internet searches
   - Arxiv Agent for academic paper searches
   - RAG Agent for document content retrieval
   - final Agent to handle all the answers provided by above agents

3. **User Interaction Flow**:
   - User authentication and session management
   - Document selection and query submission
   - Multi-agent research process
   - Answers from 3 different agents will be handled by final agent to be sent to OpenAI API
   - Response presentation and export options for PDF & Codelabs
  
4. **Export as Codelabs**:
   - Pre requisites to use codelabs export option, use below commands
     ```
     #first time installation in mac:
     curl -LO https://github.com/googlecodelabs/tools/releases/latest/download/claat-darwin-amd64
     sudo mv claat-darwin-amd64 /usr/local/bin/claat  
     chmod +x /usr/local/bin/claat 
     ```
   - You choose the Export as Codelabs option, then for opening codelabs document:
     ```
     #codelabs opening steps:
     cd downloads
     claat export .md file
     cd created dir
     claat serve
     ```
   - this will open the codelabs in localhost server

## Project Structure
```
.
├── Airflow_Data
│   ├── dags
│   ├── docker-compose.yaml
│   ├── dockerfile
│   ├── logs
│   └── requirements.txt
├── FastAPIs_Backend
│   ├── Dockerfile
│   └── main.py
├── LICENSE
├── README.md
├── Streamlit_UI
│   ├── Dockerfile
│   └── app.py
├── diagrams
│   ├── Agents_diag.py
│   ├── Assignment4_diagrams.ipynb
│   ├── Project4_diag.py
│   ├── airflow_icon.png
│   ├── arxiv_icon.png
│   ├── cfa_icon.png
│   ├── docker_icon.png
│   ├── docling_ico.png
│   ├── docling_icon.png
│   ├── fastapi_icon.png
│   ├── google_search_icon.jpg
│   ├── langgraph_agents_system.png
│   ├── langgraph_icon.png
│   ├── openai_icon.png
│   ├── pinecone_icon.png
│   ├── rag_icon.png
│   ├── research_assistant_system.png
│   ├── selenium_icon.png
│   └── streamlit_icon.png
├── docker-compose.yml
├── poc
│   ├── agent_poc1.py
│   ├── agents.ipynb
│   ├── codelab_test
│   ├── list_indexes.py
│   ├── pinecone_indexdetails.py
│   └── query.py
└── requirements.txt
```

## Setup Instructions

1. **Clone Repository**
```bash
git clone [repository-url]
cd [repository-name]
```

2. **Environment Setup**
Create necessary .env files with required variables:

```bash
# Airflow Environment
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key

# Authentication
SECRET_KEY="your key"
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
TARGET_BUCKET_NAME=docling

# FastAPI/Streamlit Environment
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

3. **Start Airflow**
```bash
cd Airflow_data
docker build
docker compose up -d
```
cd .. (back to root dir)

4. **Build and Run Application**
```bash
cd .. #back to root dir
docker compose build
docker compose up -d
```

## Deployment Guide

1. **Prepare Deployment**
```bash
# Tag images
docker tag backend:latest username/backend:latest
docker tag frontend:latest username/frontend:latest

# Push to registry
docker push username/backend:latest
docker push username/frontend:latest
```

2. **Server Setup**
```bash
# Install Docker and Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

# Create project directory
mkdir ~/research-assistant
cd ~/research-assistant

# Copy configuration files
scp -r .env credentials.json user@server:~/research-assistant/
```

3. **Deploy Application**
```bash
docker-compose pull
docker-compose up -d
```

## References

- langchain-ai.github.io/langgraph/
- docs.pinecone.io
- fastapi.tiangolo.com
- docs.streamlit.io
- platform.openai.com/docs
- docs.docker.com
