"""
For preparing data
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys
import argparse
import yaml


# Used to organize and create a unified dataframe for all datasets.
class CorpusDataFrame:

    def __init__(self):
        self.data = []
        self.exceptions = 0

    def append_file(self, path, name, label1, label2, label3):
        # Append filename, filepath, and emotion label to the data list.
        try:
            # avoid broken files
            s = torchaudio.load(path)
            self.data.append(
                {
                    "name": name,
                    "path": str(path),
                    "class_2": label1,  # For binary classification
                    "class_4": label2,  # For multiclassification
                    "score": label3,  # PHQ-8 score
                }
            )
        except Exception as e:
            print("Could not load ", str(path), e)
            self.exceptions += 1
            pass

    def data_frame(self):
        if self.exceptions > 0:
            print(f"{self.exceptions} files could not be loaded")

        # Create the dataframe from the organized data list
        df = pd.DataFrame(self.data)
        return df


# Depression list of daic-woz data for creating unified data frames
dep = [
    308,
    309,
    311,
    319,
    320,
    321,
    325,
    330,
    332,
    335,
    337,
    338,
    339,
    344,
    345,
    346,
    347,
    348,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    359,
    362,
    365,
    367,
    372,
    376,
    377,
    380,
    381,
    384,
    386,
    388,
    389,
    402,
    405,
    410,
    412,
    413,
    414,
    418,
    421,
    422,
    426,
    433,
    440,
    441,
    448,
    453,
    459,
    461,
    483,
]


def DAIC_WOZ(data_path, path_label=None):
    """
    Process DAIC-WOZ dataset files and create a dataframe.

    Args:
        data_path: Path to audio files
        path_label: Path to labels file (optional)

    Returns:
        DataFrame with processed audio information
    """
    print("PREPARING DAIC_WOZ DATA PATHS")

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        try:
            # Use pathlib for cross-platform path handling
            name = path.stem  # Gets filename without extension
            path_parts = path.parts
            filename_parts = name.split("_")

            # Extract the label from the filename
            label = filename_parts[0]
            label3 = filename_parts[1] if len(filename_parts) > 1 else "0"

            # Based on binary classification: depressed or not depressed
            if int(label) in dep:
                label1 = "dep"
            else:
                label1 = "ndep"

            # Classification of depression severity based on PHQ-8 score
            phq8_score = int(label3)
            if 0 <= phq8_score <= 4:
                label2 = "non"
            elif 5 <= phq8_score <= 9:
                label2 = "mild"
            elif 10 <= phq8_score <= 14:
                label2 = "moderate"
            else:
                label2 = "severe"

            cdf.append_file(str(path), name, label1, label2, label3)
        except (ValueError, IndexError) as e:
            print(f"Error processing file {path}: {e}")
            continue

    df = cdf.data_frame()
    return df


# Use the correct function to iterate through the named dataset.
def get_df(corpus, data_path, path_label=None):
    """
    Get DataFrame for a specific corpus

    Args:
        corpus: Name of corpus (e.g., "daic_woz")
        data_path: Path to audio data
        path_label: Path to labels file (optional)

    Returns:
        DataFrame with processed audio data
    """
    if corpus == "daic_woz":
        return DAIC_WOZ(data_path, path_label)
    else:
        raise ValueError(f"Invalid corpus name: {corpus}")


# To get the datasets names and their file paths
def df(corpora, data_path, path_label=None):
    """
    Create DataFrame from one or more corpora

    Args:
        corpora: String or list of corpus names
        data_path: String or list of paths to audio data
        path_label: Path to labels file (optional)

    Returns:
        Combined DataFrame with all corpus data
    """
    # In case more than one dataset is used.
    if isinstance(corpora, list):
        df = pd.DataFrame()
        for i, corpus in enumerate(corpora):
            df_ = get_df(corpus, data_path[i], path_label)
            df = pd.concat([df, df_], axis=0)
    else:
        df = get_df(corpora, data_path, path_label)

    print(f"Step 0: {len(df)}")

    # Filter out non-existing files
    df["status"] = df["path"].apply(lambda path: os.path.exists(path))
    df = df[df["status"] == True]
    df = df.drop("status", axis=1)
    print(f"Step 1: {len(df)}")

    # Shuffle the data
    df = df.sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)

    # Explore the label distribution
    print("Labels: ", df["class_4"].unique())
    print()
    print("Class distribution:")
    print(df.groupby("class_4").count()[["path"]])

    return df


def prepare_splits(df, config, evaluation=False):
    """
    Prepare and save train/validation/test splits

    Args:
        df: DataFrame with processed audio data
        config: Configuration dictionary
        evaluation: Whether this is for evaluation (default: False)
    """
    output_dir = config["output_dir"]
    suffix = "/eval_splits/" if evaluation else "/splits/"
    save_path = output_dir + suffix

    # Create splits directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Create train, test, and validation splits.
    random_state = config["seed"]  # 103
    # 6:2:2 split ratio
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        train_size=0.8,
        random_state=random_state,
        stratify=df["class_4"],
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.25,
        train_size=0.75,
        random_state=random_state,
        stratify=train_df["class_4"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Save each to file
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/valid.csv", sep="\t", encoding="utf-8", index=False)

    print(
        f"train: {train_df.shape},\t validate: {valid_df.shape},\t test: {test_df.shape}"
    )


# Match eval_df to df by removing additional labels in eval_df
def remove_additional_labels(df, eval_df):
    """
    Remove labels from evaluation dataset that don't exist in training dataset

    Args:
        df: Training DataFrame
        eval_df: Evaluation DataFrame

    Returns:
        Filtered evaluation DataFrame
    """
    df_labels = df["class_4"].unique()
    eval_df_labels = eval_df["class_4"].unique()

    print("Default dataset labels: ", df_labels)
    print("Evaluation dataset labels: ", eval_df_labels)

    # More efficient list comprehension
    additional_labels = [label for label in eval_df_labels if label not in df_labels]

    print("Length of evaluation dataset: \t", len(eval_df))

    # Remove labels not in the original df - cleaner boolean indexing
    eval_df = eval_df[~eval_df.class_4.isin(additional_labels)]

    print(
        f"Length of evaluation dataset after removing {additional_labels}: \t",
        len(eval_df),
    )
    eval_df_labels = eval_df["class_4"].unique()
    print("Updated evaluation dataset labels: ", eval_df_labels)

    return eval_df


if __name__ == "__main__":
    # Get the configuration file containing dataset name, path, and other configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="yaml configuration file path")
    args = parser.parse_args()
    config_file = args.config

    if not config_file:
        print("Error: Configuration file path is required")
        sys.exit(1)

    try:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Create required output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_dir"] + "/splits/", exist_ok=True)

    # Create a dataframe
    path_label = config.get("path_label")  # Get path_label if it exists
    dataset_df = df(config["corpora"], config["data_path"], path_label)

    # Create train, test, and validation splits and save them to file
    prepare_splits(dataset_df, config)

    # If a different dataset is used to test the model:
    # Uncommented and fixed the evaluation code
    if config.get("test_corpora") is not None:
        # Create a dataframe for evaluation
        eval_df = df(config["test_corpora"], config["test_corpora_path"], path_label)

        # Match eval_df to df
        eval_df = remove_additional_labels(dataset_df, eval_df)

        # Create train, test, and validation splits and save them to file
        prepare_splits(eval_df, config, evaluation=True)

    print("Data preparation complete!")
