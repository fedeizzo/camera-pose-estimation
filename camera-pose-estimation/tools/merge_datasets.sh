#!/usr/bin/env python
import argparse
import pandas as pd

from typing import List
from os.path import isfile
from os import makedirs, symlink
from pathlib import PosixPath


def is_relative_workspace(workspace: PosixPath) -> bool:
    train_file = workspace / "train_relative.csv"
    validation_file = workspace / "validation_relative.csv"
    test_file = workspace / "test_relative.csv"

    return (
        sum([isfile(train_file), isfile(validation_file), isfile(test_file)])
        == 3
    )


def create_symlinks(
    output_path: PosixPath,
    images_folder_names: List[str],
    images_folder_src: List[str],
):
    makedirs(output_path / "imgs", exist_ok=True)
    for src, dst in zip(images_folder_src, images_folder_names):
        final_path = output_path / "imgs" / dst
        symlink(PosixPath(src).resolve(), final_path)


def merge_datasets(
    output_path: PosixPath, workspaces: List[PosixPath], imgs_paths: List[str]
):
    for phase in [
        "train_relative.csv",
        "validation_relative.csv",
        "test_relative.csv",
    ]:
        files = []
        for workspace, imgs in zip(workspaces, imgs_paths):
            positions = pd.read_csv(workspace / phase)
            if isinstance(positions, pd.DataFrame):
                positions["image_t"] = positions["image_t"].apply(
                    lambda x: f"{imgs}/{x}"
                )
                positions["image_t1"] = positions["image_t1"].apply(
                    lambda x: f"{imgs}/{x}"
                )
                files.append(positions)
            else:
                raise ValueError(
                    f"Erroring reading dataset at {workspace/phase}"
                )
        merged = pd.concat(files)
        merged.to_csv(output_path / phase, index=False)


def main(args):
    workspaces = list(map(lambda x: PosixPath(x), args.input_paths))
    output_path = PosixPath(args.output_path)
    are_relative_workspaces = list(
        map(lambda x: is_relative_workspace(x), workspaces)
    )
    if sum(are_relative_workspaces) == len(workspaces):
        create_symlinks(
            output_path,
            list(map(lambda x: x.name, workspaces)),
            list(map(lambda x: x / "imgs", workspaces)),
        )
        merge_datasets(
            output_path,
            workspaces,
            list(map(lambda x: x.name, workspaces)),
        )

    else:
        raise ValueError(
            "Input paths do not correspond to relative workspaces"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to a dataset")
    parser.add_argument(
        "-i",
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="Input paths where images and models are saved",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output path where merged dataset are be saved",
    )
    args = parser.parse_args()
    main(args)
