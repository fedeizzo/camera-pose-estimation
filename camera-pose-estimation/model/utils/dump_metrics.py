import argparse
import pandas as pd

from aim import Repo
from config_parser import ConfigParser


def dump_metrics(
    config_path: str, experiment_name: str = None, run_name: str = None
) -> None:
    config = ConfigParser(config_path)
    repo = Repo(config["paths"]["aim_dir"])

    query = []
    if not run_name is None:
        query.append(f'run.environment.run_name == "{run_name}"')
    if not experiment_name is None:
        query.append(f'run.environment.experiment_name == "{experiment_name}"')

    query = " and ".join(query)

    dfs_output = []
    for run_metrics_col in repo.query_metrics(query).iter_runs():
        df_run = run_metrics_col.dataframe()
        if df_run is not None:
            dfs_output.append(df_run)

    dfs_output = pd.concat(dfs_output, ignore_index=True, sort=False)
    dfs_output.to_csv(config["paths"]["dump_metrics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file")
    parser.add_argument("-e", "--experiment_name", type=str, required=False, help="Experiment name")
    parser.add_argument("-r", "--run_name", type=str, required=False, help="Run name")

    args = parser.parse_args()
    
    dump_metrics(args.config, args.experiment_name, args.run_name)
