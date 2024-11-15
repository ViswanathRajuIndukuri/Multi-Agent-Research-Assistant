from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.programming.framework import FastAPI

# Paths to custom icons
fastapi_icon = "fastapi_icon.png"
langgraph_icon = "langgraph_icon.png"
openai_icon = "openai_icon.png"

# Diagram layout attributes
graph_attr = {
    "fontsize": "15",
    "splines": "ortho",
    "rankdir": "TB",  # Top to bottom for the agents layer
    "compound": "true",
    "nodesep": "1.2",
    "ranksep": "1.2"
}

edge_attr = {
    "color": "black",
}

# Node attributes for larger icons
large_node_attr = {
    "imagescale": "true",
    "width": "3",
    "height": "3"
}

# Create the main diagram with forced left-to-right layout
with Diagram("LangGraph Agents System", show=True, graph_attr=graph_attr, edge_attr=edge_attr, direction="LR"):
    
    # Create FastAPI (leftmost)
    fastapi = Custom("FastAPI", fastapi_icon)
    
    # Create the three agents in a vertical arrangement
    with Cluster("Primary Agents", graph_attr={"rankdir": "TB", "margin": "30"}):
        websearch_agent = Custom("Web Search Agent", langgraph_icon, **large_node_attr)
        arxiv_agent = Custom("Arxiv Agent", langgraph_icon, **large_node_attr)
        rag_agent = Custom("RAG Agent", langgraph_icon, **large_node_attr)
    
    # Create Final Agent
    final_agent = Custom("Final Agent", langgraph_icon, **large_node_attr)
    
    # Create OpenAI (rightmost)
    openai = Custom("OpenAI API", openai_icon)

    # Connections between FastAPI and agents (bidirectional)
    fastapi >> Edge(color="darkgreen") >> websearch_agent
    fastapi >> Edge(color="darkgreen") >> arxiv_agent
    fastapi >> Edge(color="darkgreen") >> rag_agent
    
    websearch_agent >> Edge(color="darkgreen") >> fastapi
    arxiv_agent >> Edge(color="darkgreen") >> fastapi
    rag_agent >> Edge(color="darkgreen") >> fastapi

    # Connections between agents and final agent (bidirectional)
    websearch_agent >> Edge(color="red") >> final_agent
    arxiv_agent >> Edge(color="red") >> final_agent
    rag_agent >> Edge(color="red") >> final_agent
    
    final_agent >> Edge(color="red") >> websearch_agent
    final_agent >> Edge(color="red") >> arxiv_agent
    final_agent >> Edge(color="red") >> rag_agent
    
    # Connection between final agent and OpenAI (bidirectional)
    final_agent >> Edge(color="blue") >> openai
    openai >> Edge(color="blue") >> final_agent