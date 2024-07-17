import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_files(file_path: str) -> str:
    with open(file_path, "r") as f:
        text = f.read()
    return text


def tokenize(text, tokenizer) -> torch.tensor:
    return tokenizer(text, return_tensors="pt")


def load_model_and_tokenizer(model_name: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_evaluation(eval_directory: str) -> dict[str, list[str]]:
    """
    Load evaluation benchmarks from a directory.
    return: A dictionary mapping evaluation names to lists of text strings.
    """
    evals = {}
    for eval_path in os.listdir(eval_directory):
        eval_name_with_extension = os.path.basename(eval_path)
        eval_name, _ = os.path.splitext(eval_name_with_extension)
        with open(f"{eval_directory}/{eval_path}", "r") as f:
            eval_benchmark = f.readlines()[1:]
            evals[eval_name] = eval_benchmark
    return evals


def write_output_to_file(output_path, text):
    with open(output_path, "a") as f:
        f.write(text + "\n")
