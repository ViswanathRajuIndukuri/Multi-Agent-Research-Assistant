import os
import time
from dotenv import load_dotenv

from serpapi.google_search import GoogleSearch

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

from pinecone import Pinecone, ServerlessSpec

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI


from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.graph import StateGraph, MessagesState, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Load environment variables from .env file
load_dotenv()

# Verify that API keys are loaded
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

# Debugging: Print values to confirm they are loaded (useful for troubleshooting)
# print("Pinecone API Key:", pinecone_api_key)
# print("OpenAI API Key:", openai_api_key)
# print("SERPAPI API Key:", serpapi_key)

# Optional: Check if keys are missing
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found in environment variables.")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")
if not serpapi_key:
    raise ValueError("SERPAPI API Key not found in environment variables.")

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
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return web_search_results

## Arxiv
arxiv_wrapper=ArxivAPIWrapper(top_k_results=5)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv_tool.invoke("what is investment model validation")

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


# Configure client
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
retriever = vector_store.as_retriever(search_kwargs={'k': 2})

def create_rag_agent():
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
def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("web_agent", create_web_agent())
    workflow.add_node("arxiv_agent", create_arxiv_agent())
    workflow.add_node("rag_agent", create_rag_agent())
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

# Modified query function
def run_research_query(query: str):
    try:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "web_results": "",
            "arxiv_results": "",
            "rag_results": ""
        }
        
        final_state = research_app.invoke(initial_state)
        return final_state["messages"][-1].content
    except Exception as e:
        print(f"Query execution error: {str(e)}")
        return "An error occurred while processing your query. Please try again."

# Create research application
research_app = build_workflow()

# Example usage
if __name__ == "__main__":
    query = "What are the best practices for investment model validation?"
    response = run_research_query(query)
    print("\nQuery Results:")
    print("=" * 80)
    print(response)
    print("=" * 80)