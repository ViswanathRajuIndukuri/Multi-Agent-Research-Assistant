# Assignment4

Airflow data pipeline:

downloadFiles.py will download the 3 PDFs
process_Files.py will use docling to process the store them in pinecone
dag file will run them in pipeline
dockerfile and docker-compose.yaml to support airflow in docker

clone the repo

cd Airflow

pip intall -r requirements.txt

add .env file in Airflow folder with below vars

```
AIRFLOW_UID=502

OENAI_API_KEy="your key"
PINECONE_API_KEY="your key"
```
done