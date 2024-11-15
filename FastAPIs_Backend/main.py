# fastapi_app.py

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from psycopg2 import connect, sql, OperationalError, errors
from uuid import uuid4, UUID
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2.service_account import Credentials
from contextlib import asynccontextmanager
import io
import time
from serpapi.google_search import GoogleSearch
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from pinecone import Pinecone, ServerlessSpec
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from md2pdf.core import md2pdf
import logging
import markdown2
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gcp_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_creds_path

# Initialize Pinecone using the Pinecone class
pc = Pinecone(api_key=pinecone_api_key)

# Initialize Google Cloud Storage client with credentials
if gcp_creds_path:
    credentials = Credentials.from_service_account_file(gcp_creds_path)
    storage_client = storage.Client(credentials=credentials)
else:
    raise RuntimeError("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS in .env")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

CREATE_USERS_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

# FastAPI lifespan event to initialize resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Google Cloud Storage client with credentials
    if gcp_creds_path:
        credentials = Credentials.from_service_account_file(gcp_creds_path)
        app.state.storage_client = storage.Client(credentials=credentials)
    else:
        raise RuntimeError("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS in .env")
    app.state.pinecone_client = Pinecone(api_key=pinecone_api_key)
    # Initialize database
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(CREATE_USERS_TABLE_QUERY)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {e}")
    finally:
        cur.close()
        conn.close()
    
    yield  # Execute application startup and shutdown processes

app = FastAPI(lifespan=lifespan)

# Database connection dependency with error handling
def get_db_connection():
    try:
        return connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except OperationalError:
        raise HTTPException(status_code=500, detail="Database connection error")


def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, hashed_password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if user and verify_password(password, user[1]):
            return {"id": user[0], "username": username}
        return None
    except Exception:
        raise HTTPException(status_code=500, detail="Error during authentication")
    finally:
        cur.close()
        conn.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Pydantic models with input validation
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        if len(v) < 3 or len(v) > 30:
            raise ValueError('Username must be between 3 and 30 characters')
        return v

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateReportRequest(BaseModel):
    chat_history: List[ChatMessage]

class ResearchQuery(BaseModel):
    query: str
    index_name: str

@app.post("/register", response_model=UserOut)
def register_user(user: UserCreate):
    try:
        hashed_password = hash_password(user.password)
        user_id = uuid4()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (id, email, username, hashed_password) VALUES (%s, %s, %s, %s) RETURNING created_at",
            (str(user_id), user.email, user.username, hashed_password))
        created_at = cur.fetchone()[0]
        conn.commit()
        return {"id": user_id, "email": user.email, "username": user.username, "created_at": created_at}
    except errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Email or username already exists")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Error during registration")
    finally:
        cur.close()
        conn.close()


@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not form_data.username or not form_data.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user['username']}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserOut)
async def read_user_me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, username, created_at FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": UUID(user[0]), "email": user[1], "username": user[2], "created_at": user[3]}
    except Exception:
        raise HTTPException(status_code=500, detail="Error retrieving user information")
    finally:
        cur.close()
        conn.close()

@app.get("/pinecone-indexes")
def list_pinecone_indexes(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        index_list = pc.list_indexes().names()
        return {"indexes": index_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

serpapi_params = {
    "engine": "google",
    "api_key": serpapi_key
}

@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5
    })
    results = search.get_dict()["organic_results"]
    web_search_results = "\n---\n".join(
        ["\n".join([x.get("title", ""), x.get("snippet", ""), x.get("link", "")]) for x in results]
    )
    return web_search_results

## Arxiv
arxiv_wrapper = ArxivAPIWrapper(top_k_results=5)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

@tool("search_arxiv")
def search_arxiv(query: str):
    """
    Searches Arxiv for academic papers based on the query provided.
    Returns a formatted list of the top results.
    """
    results = arxiv_tool.invoke(query)
    
    # Split results by "Published:" to separate each paper entry, excluding the first empty split
    entries = results.split("Published: ")[1:]
    
    # Format each entry
    arxiv_search_results = "\n\n---\n\n".join(
        [f"Published: {entry.strip()}" for entry in entries]
    )
    return arxiv_search_results

# Define state schema without 'next' since we're running in parallel
class AgentState(MessagesState):
    web_results: str
    arxiv_results: str
    rag_results: str

# Modified agent functions without 'next' state
def create_web_agent():
    def web_agent(state: AgentState):
        try:
            query = state["messages"][0].content
            results = web_search.invoke(query)
            if not results:
                results = "No relevant web search results found."
            return {
                "web_results": results
            }
        except Exception as e:
            print(f"Web agent error: {str(e)}")
            return {
                "web_results": "Error in web search."
            }
    return web_agent

def create_arxiv_agent():
    def arxiv_agent(state: AgentState):
        try:
            query = state["messages"][0].content
            results = search_arxiv.invoke(query)
            if not results:
                results = "No relevant academic papers found."
            return {
                "arxiv_results": results
            }
        except Exception as e:
            print(f"arXiv agent error: {str(e)}")
            return {
                "arxiv_results": "Error in academic search."
            }
    return arxiv_agent

def create_rag_agent(index_name):
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})
    
    def rag_agent(state: AgentState):
        try:
            query = state["messages"][0].content
            docs = retriever.invoke(query)
            results = "\n".join([doc.page_content for doc in docs])
            if not results:
                results = "No relevant documents found in the knowledge base."
            return {
                "rag_results": results
            }
        except Exception as e:
            print(f"RAG agent error: {str(e)}")
            return {
                "rag_results": "Error in document retrieval."
            }
    return rag_agent

def create_final_agent():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    def final_agent(state: AgentState):
        system_prompt = """You are an expert research assistant tasked with synthesizing information from multiple sources.
Your goal is to provide a comprehensive, well-structured response that:
1. Combines insights from web searches, academic research, and internal documents
2. Organizes information logically with clear sections
3. Highlights key findings and practical implications
4. Maintains academic rigor while being accessible
5. Provides specific examples and evidence when available"""

        user_query = state["messages"][0].content
        
        analysis_prompt = f"""
Based on the following information sources:

WEB SEARCH FINDINGS:
{state['web_results']}

ACADEMIC RESEARCH:
{state['arxiv_results']}

INTERNAL DOCUMENT ANALYSIS:
{state['rag_results']}

Please provide a comprehensive analysis addressing this query:
{user_query}

Format your response with:
1. Key Findings
2. Detailed Analysis
3. Practical Implications
4. Recommendations (if applicable)
5. Web Source links
6. Arxiv Sources

Ensure to synthesize information across all sources and highlight any important patterns or contradictions."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            return {
                "messages": [*state["messages"], AIMessage(content=response.content)]
            }
        except Exception as e:
            print(f"Final agent error: {str(e)}")
            return {
                "messages": [*state["messages"], 
                           AIMessage(content="Error in generating final response.")]
            }
    
    return final_agent

# Modified workflow for parallel execution
def build_workflow(index_name):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("web_agent", create_web_agent())
    workflow.add_node("arxiv_agent", create_arxiv_agent())
    workflow.add_node("rag_agent", create_rag_agent(index_name))
    workflow.add_node("final", create_final_agent())
    
    # Add edges for parallel execution
    workflow.add_edge("web_agent", "final")
    workflow.add_edge("arxiv_agent", "final")
    workflow.add_edge("rag_agent", "final")
    workflow.add_edge("final", END)
    
    # Set multiple entry points for parallel execution
    workflow.set_entry_point("web_agent")
    workflow.set_entry_point("arxiv_agent")
    workflow.set_entry_point("rag_agent")
    
    return workflow.compile()

@app.post("/run-research-query")
def run_research_query_endpoint(
    research_query: ResearchQuery,
    token: str = Depends(oauth2_scheme)
):
    # Authentication logic within the endpoint
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = research_query.query
    index_name = research_query.index_name

    # Build the workflow
    research_app = build_workflow(index_name)

    # Run the research query
    try:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "web_results": "",
            "arxiv_results": "",
            "rag_results": ""
        }
        final_state = research_app.invoke(initial_state)
        response_content = final_state["messages"][-1].content
        return {"result": response_content}
    except Exception as e:
        print(f"Query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

# Updated generate_report endpoint using WeasyPrint
@app.post("/generate-report")
def generate_report(
    token: str = Depends(oauth2_scheme),
    request: GenerateReportRequest = Body(...)
):
    # Authenticate the user
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Generate the report
    try:
        chat_history = request.chat_history

        # Combine the chat history into a markdown string
        md_content = ""
        for message in chat_history:
            role = message.role
            content = message.content
            if role == "user":
                md_content += f"### Question:\n{content}\n\n"
            elif role == "assistant":
                md_content += f"### Research Results:\n{content}\n\n"

        # Log the md_content
        logger.info(f"Markdown content:\n{md_content}")

        # Convert markdown to HTML
        html_content = markdown2.markdown(md_content)

        # Generate PDF from HTML using WeasyPrint
        pdf_bytes = HTML(string=html_content).write_pdf()

        # Return the PDF as a StreamingResponse
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename=report.pdf"
        })
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the report: {str(e)}")
    

class Message(BaseModel):
    role: str
    content: str

class GenerateReportRequest(BaseModel):
    chat_history: List[Message]

@app.post("/generate-markdown-codelab")
def generate_markdown_codelab(
    token: str = Depends(oauth2_scheme),
    request: GenerateReportRequest = Body(...)
):
    # Authenticate the user
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Generate the Codelab content in Markdown format
    try:
        chat_history = request.chat_history

        # Codelab metadata and introductory section
        md_content = """# Interactive Research Session
summary: Generated research session report in Codelab format.
id: research-session-codelab
categories: Research

## Introduction
This Codelab presents the research session in a question-answer format, based on user queries and the assistant's responses.
"""

        # Process chat history, pairing each user question with assistant's response
        question_count = 1
        for i in range(0, len(chat_history) - 1, 2):
            user_message = chat_history[i].content if chat_history[i].role == "user" else ""
            assistant_message = chat_history[i + 1].content if chat_history[i + 1].role == "assistant" else ""
            
            md_content += f"\n## Question {question_count}\n\n{user_message}\n\n"
            md_content += f"### Response\n\n{assistant_message}\n\n"
            question_count += 1

        # Log the md_content
        logger.info(f"Codelab Markdown content:\n{md_content}")

        # Return the Codelab-compatible Markdown file as a StreamingResponse
        return StreamingResponse(io.BytesIO(md_content.encode()), media_type="text/markdown", headers={
            "Content-Disposition": f"attachment; filename=research_codelab.md"
        })

    except Exception as e:
        logger.error(f"Error generating Codelab markdown file: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the Codelab markdown file: {str(e)}")