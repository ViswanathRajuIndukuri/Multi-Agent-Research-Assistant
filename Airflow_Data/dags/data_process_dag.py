from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'download_and_process_pdfs_docling',
    default_args=default_args,
    description='Download PDFs and process them',
    schedule_interval=None,  # Set to desired schedule
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    download_pdfs = BashOperator(
        task_id='download_pdfs',
        bash_command='python /opt/airflow/dags/downloadFiles.py',
    )

    process_pdfs = BashOperator(
        task_id='process_pdfs',
        bash_command='python /opt/airflow/dags/process_Files.py',
    )

    download_pdfs >> process_pdfs