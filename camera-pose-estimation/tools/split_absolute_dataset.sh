#!/usr/bin/env python
import argparse
import pickle

import pandas as pd

from pathlib import PosixPath
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler


def split(
    positions: pd.DataFrame,
    train_split: Optional[int],
    validation_split: Optional[int],
    test_split: Optional[int],
):
    train_split = train_split if train_split else int(len(positions) / 3)
    validation_split = (
        validation_split + train_split
        if validation_split
        else int(len(positions) / 3) + train_split
    )
    test_split = (
        test_split + validation_split
        if test_split
        else int(len(positions) / 3) + validation_split
    )

    print("Splits are:")
    print(f"\ttrain: [0-{train_split}[")
    print(f"\tvalidation: [{train_split}-{validation_split}[")
    print(f"\ttest: [{validation_split}-{test_split}[")

    train = positions.iloc[0::3]
    validation = positions.iloc[1::3]
    test = positions.iloc[2::3]

    # train = positions.iloc[:train_split]
    # validation = positions.iloc[train_split:validation_split]
    # test = positions.iloc[validation_split:test_split]
    

    return train, validation, test


def normalize(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    scaler_path: PosixPath,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quaternion_scaler = MinMaxScaler()
    quaternion_scaler = quaternion_scaler.fit(
        train[["qx", "qy", "qz", "qw"]].values.flatten().reshape(-1, 1)
    )
    translation_scaler = MinMaxScaler()
    translation_scaler = translation_scaler.fit(
        train[["tx", "ty", "tz"]].values.flatten().reshape(-1, 1)
    )

    for phase in [train, validation, test]:
        for col in ["qx", "qy", "qz", "qw"]:
            phase.update(
                {
                    col: quaternion_scaler.transform(
                        phase[col].values.reshape(-1, 1)
                    ).flatten()
                }
            )

        for col in ["tx", "ty", "tz"]:
            phase.update(
                {
                    col: translation_scaler.transform(
                        phase[col].values.reshape(-1, 1)
                    ).flatten()
                }
            )

    for scaler, filepath in zip(
        [quaternion_scaler, translation_scaler],
        [
            scaler_path / "quaternion_scaler.pkl",
            scaler_path / "translation_scaler.pkl",
        ],
    ):
        with open(filepath, "wb") as f:
            pickle.dump(scaler, f)

    return train, validation, test


def main(args):
    input_path = PosixPath(args.input_path)
    positions_file = input_path / "positions.csv"
    positions = pd.read_csv(positions_file)
    if isinstance(positions, pd.DataFrame):
        print(f"There are {len(positions)} positions")
        train, validation, test = split(
            positions, args.train_split, args.validation_split, args.test_split
        )
        if not args.no_normalization:
            print("Normalizing data...")
            train, validation, test = normalize(
                train.copy(deep=True),
                validation.copy(deep=True),
                test.copy(deep=True),
                input_path,
            )
        train_file = input_path / "train.csv"
        validation_file = input_path / "validation.csv"
        test_file = input_path / "test.csv"
        train.to_csv(train_file, index=False)
        validation.to_csv(validation_file, index=False)
        test.to_csv(test_file, index=False)
    else:
        raise ValueError(f"Erroring reading {positions_file} file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to a dataset")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Input path where images and models are saved",
    )
    parser.add_argument(
        "-n",
        "--no_normalization",
        action="store_true",
        help="Block normalization process",
    )
    parser.add_argument(
        "-t",
        "--train_split",
        type=int,
        required=False,
        help="Number of samples for train phase",
    )
    parser.add_argument(
        "-v",
        "--validation_split",
        type=int,
        required=False,
        help="Number of samples for validation phase",
    )
    parser.add_argument(
        "-e",
        "--test_split",
        type=int,
        required=False,
        help="Number of samples for test phase",
    )
    args = parser.parse_args()
    main(args)
