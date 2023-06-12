import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts

def log_experimento(host, nombre_experimento="default", run_name="", descripcion="", dataset_tag="validacion", parametros={}, metricas={}, model=None, artifact_path=""):
    mlflow.set_tracking_uri(host)
    mlflow.set_experiment(nombre_experimento)
    experiment = mlflow.get_experiment_by_name(nombre_experimento)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, description=descripcion):
        mlflow.set_tag("dataset", dataset_tag)
        
        for key, value in parametros.items():
            mlflow.log_param(key, value)
            
        for key, value in metricas.items():
            mlflow.log_metric(key, value)
    
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path
        )
        
        mlflow.end_run()