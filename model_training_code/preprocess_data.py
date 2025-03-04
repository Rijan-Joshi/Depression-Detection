"""
Preprocess data
"""

import sys
import os
import argparse
import yaml
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, Wav2Vec2Processor
from nested_array_catcher import nested_array_catcher


# Label sorting specialized for daic-woz quadruple categorization
def custom_sort(ls):
    sort_rule = [("non", 0), ("mild", 1), ("moderate", 2), ("severe", 3)]
    sort_ls = []
    for i in ls:
        for rule in sort_rule:
            if rule[0] in i:
                sort_ls.append((rule[1], i))
                break
    sort_ls.sort()
    return [i[1] for i in sort_ls]


# for training
def training_data(configuration):
    # Create cache directory if it doesn't exist
    cache_dir = configuration.get("cache_dir", "./content/cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load the created dataset splits using datasets
    train_filepath = configuration["output_dir"] + "/splits/train.csv"
    valid_filepath = configuration["output_dir"] + "/splits/valid.csv"

    print(f"Loading train data from: {train_filepath}")
    print(f"Loading validation data from: {valid_filepath}")

    # Verify that files exist
    if not os.path.exists(train_filepath):
        raise FileNotFoundError(f"Train file not found: {train_filepath}")
    if not os.path.exists(valid_filepath):
        raise FileNotFoundError(f"Validation file not found: {valid_filepath}")

    # Load data directly without using cache
    try:
        train_df = pd.read_csv(train_filepath, sep="\t")
        valid_df = pd.read_csv(valid_filepath, sep="\t")

        # Create datasets from dataframes
        from datasets import Dataset

        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(valid_df)
    except Exception as e:
        print(f"Error loading CSV files directly: {e}")

        # Fallback to using load_dataset
        data_files = {
            "train": train_filepath,
            "validation": valid_filepath,
        }

        dataset = load_dataset(
            "csv",
            data_files=data_files,
            delimiter="\t",
            cache_dir=cache_dir,
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    # Specify the input and output columns
    input_column = "path"
    output_column = "class_2"  # For binary classification
    # output_column = "class_4"  # For multiple classifications

    # Distinguish the unique labels in the dataset
    label_list = train_dataset.unique(output_column)
    num_labels = len(label_list)
    if num_labels == 2:
        label_list.sort()  # Label Sorting
    else:
        label_list = custom_sort(label_list)  # For 4-category label sorting
    print(f"A classification problem with {num_labels} classes: {label_list}")

    return (
        train_dataset,
        eval_dataset,
        input_column,
        output_column,
        label_list,
        num_labels,
    )


def load_processor(configuration, label_list, num_labels=None):
    # Handle the case when num_labels is not provided
    if num_labels is None:
        num_labels = len(label_list)

    cache_dir = configuration.get("cache_dir", "./content/cache")
    os.makedirs(cache_dir, exist_ok=True)

    processor_name_or_path = configuration["processor_name_or_path"]
    pooling_mode = configuration["pooling_mode"]

    # Load model configuration
    config = AutoConfig.from_pretrained(
        processor_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        cache_dir=cache_dir,
    )
    setattr(config, "pooling_mode", pooling_mode)

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(
        processor_name_or_path, cache_dir=cache_dir
    )
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The sampling rate: {target_sampling_rate}")

    return config, processor, target_sampling_rate


def preprocess_data(
    configuration,
    processor,
    target_sampling_rate,
    train_dataset,
    eval_dataset,
    input_column,
    output_column,
    label_list,
):
    def speech_file_to_array_fn(path):
        try:
            # Check if file exists
            if not os.path.exists(path):
                print(f"Warning: Audio file not found: {path}")
                return np.zeros(16000, dtype=np.float32)

            speech_array, sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(
                sampling_rate, target_sampling_rate
            )
            speech = resampler(speech_array).squeeze().numpy()
            return speech
        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
            # Return a small empty array as a fallback
            return np.zeros(16000, dtype=np.float32)

    def label_to_id(label, label_list):
        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1
        return label

    def preprocess_function(examples):
        # Read all the audio files and resample them to 16kHz
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
        # Map each audio file to the corresponding label
        target_list = [
            label_to_id(label, label_list) for label in examples[output_column]
        ]

        # Process the audio
        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        print("\nASSERTING dtype")
        for i in tqdm(range(len(result["input_values"]))):
            result["input_values"][i] = nested_array_catcher(result["input_values"][i])

        return result

    # Create features directory if it doesn't exist
    features_path = os.path.join(configuration["output_dir"], "features")
    os.makedirs(features_path, exist_ok=True)

    train_features_path = os.path.join(features_path, "train_dataset")
    eval_features_path = os.path.join(features_path, "eval_dataset")

    if os.path.exists(train_features_path) and os.path.exists(eval_features_path):
        # Load preprocessed datasets from file
        try:
            train_dataset = load_from_disk(train_features_path)
            eval_dataset = load_from_disk(eval_features_path)
            print("Loaded preprocessed dataset from file")
        except Exception as e:
            print(f"Error loading preprocessed datasets: {e}")
            print("Will reprocess the data...")
            # Continue to reprocess

    if not os.path.exists(train_features_path) or not os.path.exists(
        eval_features_path
    ):
        # Preprocess features using a multiprocess map function
        print("Processing training dataset...")
        train_dataset = train_dataset.map(
            preprocess_function, batch_size=100, batched=True, num_proc=4
        )

        # Save preprocessed dataset to file
        try:
            train_dataset.save_to_disk(train_features_path)
            print(f"Training features saved to {train_features_path}")
        except Exception as e:
            print(f"Error saving training features: {e}")

        print("Processing evaluation dataset...")
        eval_dataset = eval_dataset.map(
            preprocess_function, batch_size=100, batched=True, num_proc=4
        )

        try:
            eval_dataset.save_to_disk(eval_features_path)
            print(f"Evaluation features saved to {eval_features_path}")
        except Exception as e:
            print(f"Error saving evaluation features: {e}")

    print("train_dataset: ", train_dataset)
    idx = 0

    # Check for return_attention_mask in configuration
    return_attention_mask = configuration.get("return_attention_mask", False)
    print("return_attention_mask:\t", return_attention_mask)

    if return_attention_mask and "attention_mask" in train_dataset[idx]:
        print(f"Training attention_mask: {train_dataset[idx]['attention_mask'][0]}")

    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="yaml configuration file path")
    args = parser.parse_args()
    config_file = args.config

    if not config_file:
        print("Error: Configuration file path is required")
        import sys

        sys.exit(1)

    try:
        with open(config_file) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        import sys

        sys.exit(1)

    # Make sure cache_dir exists
    cache_dir = configuration.get("cache_dir", "./content/cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    # prepare_data
    output_dir = configuration["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_filepath = os.path.join(splits_dir, "train.csv")
    test_filepath = os.path.join(splits_dir, "test.csv")
    valid_filepath = os.path.join(splits_dir, "valid.csv")

    if (
        not os.path.exists(train_filepath)
        or not os.path.exists(test_filepath)
        or not os.path.exists(valid_filepath)
    ):
        import prepare_data

        # prepare datasplits
        print("Data splits not found. Creating splits...")
        try:
            df = prepare_data.df(configuration["corpora"], configuration["data_path"])
            prepare_data.prepare_splits(df, configuration)
            print("Data splits created successfully")
        except Exception as e:
            print(f"Error creating data splits: {e}")
            sys.exit(1)

    # Import pandas for CSV loading
    import pandas as pd

    try:
        # Preprocess data
        print("Starting data preprocessing...")
        (
            train_dataset,
            eval_dataset,
            input_column,
            output_column,
            label_list,
            num_labels,
        ) = training_data(configuration)
        config, processor, target_sampling_rate = load_processor(
            configuration, label_list, num_labels
        )
        train_dataset, eval_dataset = preprocess_data(
            configuration,
            processor,
            target_sampling_rate,
            train_dataset,
            eval_dataset,
            input_column,
            output_column,
            label_list,
        )
        print("Preprocessing complete!")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error during preprocessing: {e}")
        sys.exit(1)
