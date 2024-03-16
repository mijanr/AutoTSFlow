"""
We will generate a table here with the results of the experiments from mlruns
"""

import mlflow

def get_best_runs():  

    mlflow.set_tracking_uri("file:../mlruns")
    all_runs = mlflow.search_runs(search_all_experiments=True)
    columns = ['tags.mlflow.runName', 'params.model_name', 'metrics.accuracy']
    all_runs = all_runs[columns]
    best_runs = all_runs.groupby(['tags.mlflow.runName', 'params.model_name']).agg({'metrics.accuracy': 'max'}).reset_index()
    best_runs = best_runs.pivot(index='tags.mlflow.runName', columns='params.model_name', values='metrics.accuracy').reset_index()
    best_runs.columns.name = None
    best_runs = best_runs.rename(columns={'tags.mlflow.runName': 'Dataset'})
    # save as markdown file
    with open('best_runs.md', 'w') as f:
        f.write(best_runs.to_markdown(index=False))


if __name__ == "__main__":
    get_best_runs()
    