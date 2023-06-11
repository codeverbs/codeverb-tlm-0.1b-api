import shutil, subprocess, requests
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from git import Repo

venv_path = '/home/malik/Desktop/MLOps/model-api/venv37'  # Specify the path to the new virtual environment
python_version = '3.7'  # Specify the desired Python version
activate_command = f"source {venv_path}/bin/activate" # Command to activate the virtual environment

# Clone the GitHub repository
def clone_github_repo():
    repo_url = "https://github.com/codeverbs/codeverb-tlm-0.1b-api"
    local_dir = "/home/malik/Desktop/MLOps/model-api"
    shutil.rmtree(local_dir)
    Repo.clone_from(repo_url, local_dir)

# Create a new virtual environment and install dependencies
def create_env_and_build():
    venv_command = f"python{python_version} -m venv {venv_path}"
    subprocess.run(venv_command, shell=True, check=True)
    pip_command = f"{venv_path}/bin/pip install -r /home/malik/Desktop/MLOps/model-api/requirements.txt"
    subprocess.run(f"{activate_command} && {pip_command}", shell=True, check=True)
    subprocess.run(f"{activate_command} && python --version", shell=True, check=True)  # Verify Python version
    subprocess.run("deactivate", shell=True, check=True)

# Task to run your AI model
def run_model():
    api_command = f"python{python_version} /home/malik/Desktop/MLOps/model-api/app.py"
    subprocess.run(f"{activate_command} && {api_command}", shell=True, check=True)

# Task to run predictions on test data
def run_tests():
    test_cases = [
        {
            'prompt': 'Sort a list of numbers',
            'expected_output': 'sorted_list = sorted(my_list)'
        },
        {
            'prompt': 'Function to add two number',
            'expected_output': 'sum(a,b):\n\treturn a+b'
        },
        # Add more test cases as needed
    ]

    # Make requests to the model API and check the predictions
    for test_case in test_cases:
        prompt = test_case['prompt']
        expected_output = test_case['expected_output']
        # Make a request to the model API
        api_endpoint = 'http://localhost:5050/api/predict'
        response = requests.post(api_endpoint, json={'query': prompt, 'model': 'CodeVerbTLM-0.1B', 'inference_type': 'Comment2Python'})
        if response.status_code == 200:
            prediction = response.json().get('result')
            print(f'Prompt: {prompt}')
            print(f'Expected Output: {expected_output}')
            print(f'Prediction: {prediction}')
            print('-' * 50)
            # Perform assertion or comparison with the expected output
            assert prediction == expected_output
        else:
            # return error to Airflow
            raise Exception(f'Error occurred while making API request for prompt: {prompt}')


# Define the DAG
dag = DAG(
    'ai_model_testing',
    description='DAG to test AI large language code prediction model',
    schedule_interval=None,  # Set the schedule interval as needed
    start_date=datetime(2023, 6, 11),
    catchup=False
)

# Define the tasks
clone_model_task = PythonOperator(
    task_id='clone_model',
    python_callable=clone_github_repo,
    dag=dag
)

create_env_task = PythonOperator(
    task_id='create_env',
    python_callable=create_env_and_build,
    dag=dag
)

run_model_task = PythonOperator(
    task_id='run_model',
    python_callable=run_model,
    dag=dag
)

run_tests_task = PythonOperator(
    task_id='run_tests',
    python_callable=run_tests,
    dag=dag
)

# Define the task dependencies
# clone_model_task >> create_env_task
clone_model_task >> create_env_task >> run_model_task >> run_tests_task
