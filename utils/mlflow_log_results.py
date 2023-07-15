import mlflow

def log_results(cfg, results):
    """
    Log results with mlflow

    Parameters
    ----------
    cfg : DictConfig
        Hydra config
    results : dict
        Dictionary containing results
    """
    model_name = cfg.models.model._target_.split('.')[-1].replace("_classifier", "")
    #set experiment name with model name
    mlflow.set_experiment(model_name)
    #run name is the dataset name
    dataset_name = cfg.dataset_name[0]
    with mlflow.start_run(run_name=cfg.dataset_name[0]):
        #log metric
        mlflow.log_metric("accuracy", results['accuracy'])
        mlflow.log_metric("f1", results["cl_report"]["weighted avg"]["f1-score"])
        mlflow.log_metric("precision", results["cl_report"]["macro avg"]["precision"])
        mlflow.log_metric("recall", results["cl_report"]["macro avg"]["recall"])
        mlflow.log_metric("support", results["cl_report"]["macro avg"]["support"])
        mlflow.log_metric("macro_f1_score", results["cl_report"]["macro avg"]["f1-score"])

        #log artifacts
        mlflow.log_dict(results['cl_report'], "classification_report.json")

        #log model params
        model_params = dict(cfg.models.model)
        del model_params['_target_']
        mlflow.log_dict(model_params, "model_params.json")

        #log the classifier name 
        mlflow.log_param("model_name", model_name)

        #log epochs and learning rate
        mlflow.log_param("epochs", cfg.training_params.epochs)
        mlflow.log_param("lr", cfg.training_params.lr)

        #stop mlflow run
        mlflow.end_run()

        return 



