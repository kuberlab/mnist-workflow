import datetime

import airflow
from airflow.operators import python_operator as python
from mlboardclient.api import client


dag = airflow.DAG(
    'mnist_serve',
    description='Start mnist serving',
    catchup=False,
    schedule_interval=None,
    start_date=datetime.datetime(2018, 3, 27),
    max_active_runs=4,
)


def run_task(*args, **kwargs):
    print(kwargs)

    # While dag is run via
    # airflow trigger_dag <dag> --conf '{"conf_param": "value"}'
    # In kwargs conf can be accessed
    # kwargs['dag_run'].conf['conf_param'] == "value"

    m = client.Client(
        workspace_id='21',
        workspace_name='kuberlab-demo',
        project_name='mnist-workflow1'
    )

    app = m.apps.get()
    serving = app.servings[0]

    task = 'train'
    build = '22'

    serving.start(task, build)


run_task_op = python.PythonOperator(
    python_callable=run_task,
    provide_context=True,
    task_id='run_mnist_serve',
    dag=dag
)
