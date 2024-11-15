from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.custom import Custom
from diagrams.programming.framework import FastAPI
from diagrams.onprem.container import Docker
from diagrams.gcp.compute import GCE
from diagrams.gcp.database import SQL
from diagrams.gcp.storage import Storage as GCS

# Paths to custom icons
streamlit_icon = "streamlit_icon.png"
openai_icon = "openai_icon.png"
docker_icon = "docker_icon.png"
fastapi_icon = "fastapi_icon.png"
airflow_icon = "airflow_icon.png"
docling_icon = "docling_icon.png"
pinecone_icon = "pinecone_icon.png"
selenium_icon = "selenium_icon.png"
cfa_icon = "cfa_icon.png"
langgraph_icon = "langgraph_icon.png"

# Diagram layout attributes
graph_attr = {
    "fontsize": "15",
    "splines": "ortho",
    "rankdir": "LR",
    "compound": "true"
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

# Create the main diagram
with Diagram("Research Assistant System", show=True, graph_attr=graph_attr, edge_attr=edge_attr):
    # Data Ingestion Layer
    with Cluster("Data Ingestion Layer"):
        cfa_website = Custom("CFA Website", cfa_icon)
        selenium = Custom("Selenium", selenium_icon)
        docling = Custom("Docling", docling_icon)  # Moved docling here
        airflow = Custom("Airflow", airflow_icon)
        
        cfa_website >> selenium >> docling >> airflow  # Updated flow to include docling

    # Storage Layer
    with Cluster("Storage Layer"):
        gcs = GCS("Google Cloud Storage")
        pinecone = Custom("Pinecone Vector Store", pinecone_icon)
            
        airflow >> [gcs, pinecone]

    # Frontend Layer
    with Cluster("Frontend Layer"):
        user = User("End User")
        streamlit = Custom("Streamlit", streamlit_icon)

    # Processing & Backend Layer
    with Cluster("Processing & Backend Layer"):
        with Cluster("Docker Deployment", graph_attr={"labeljust": "r"}):
            docker_compose = Custom("Docker Compose", docker_icon)
            docker_fastapi = Custom("FastAPI Container", docker_icon)
            docker_streamlit = Custom("Streamlit Container", docker_icon)
            gcp_vm = GCE("GCP VM")
            
            [docker_fastapi, docker_streamlit] >> docker_compose >> gcp_vm

        fastapi = Custom("FastAPI", fastapi_icon)
        gcloud_postgres = SQL("GCloud PostgreSQL")  # Added GCloud PostgreSQL in place of docling
        langgraph = Custom("LangGraph", langgraph_icon, **large_node_attr)
        openai = Custom("OpenAI API", openai_icon)

        # Processing connections
        fastapi >> docker_fastapi

        # Bidirectional connections with arrows
        fastapi >> Edge(color="black") >> langgraph
        langgraph >> Edge(color="black") >> openai
        openai >> Edge(color="black") >> langgraph
        langgraph >> Edge(color="black") >> fastapi
        
        # Updated connections to use gcloud_postgres instead of docling
        fastapi >> Edge(color="black") >> gcloud_postgres
        gcloud_postgres >> Edge(color="black") >> fastapi
        
        fastapi >> Edge(color="black") >> gcs
        gcs >> Edge(color="black") >> fastapi
        
        
        
        fastapi >> Edge(color="black") >> pinecone
        pinecone >> Edge(color="black") >> fastapi

    # Frontend connections
    user >> streamlit
    streamlit >> docker_streamlit
    fastapi >> streamlit
    streamlit >> Edge(color="black") >> fastapi