import random

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_evaluation


def create_answers(model, tokenizer, context) -> list[str]:
    inputs = tokenizer(context, return_tensors="pt")
    outputs = [context]

    for _ in range(3):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50)
        outputs.append(
            tokenizer.decode(out[0], skip_special_tokens=True).replace(context, "")
        )
    return outputs


def create_context(
    model, tokenizer, eval_benchmark: list[str], system_prompt: str
) -> list[list[str]]:
    answer_list = []
    for message in tqdm(eval_benchmark):
        if message.startswith("user"):
            prompt = system_prompt + "\n" + message
        if message.startswith("assistant"):
            answers = create_answers(model, tokenizer, prompt)
            random_idx = random.randint(1, 4)
            answers.insert(random_idx, message)
            answers.append(random_idx)
            answer_list.append(answers)
    return answer_list


if __name__ == "__main__":
    device = "mps"
    model_name = "google/gemma-2b-it"

    eval_directory = "eval"

    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    minimal_prompt = """
    Your name is Ash. You are a counselor on a phone call.
    Rules:
    - You always speak in a professional manner.
    - You always use perfect grammar and spelling.
    - You vary the length and structure of your responses to sound natural."""

    evals = load_evaluation(eval_directory)

    answers = []
    for eval_name, eval_benchmark in evals.items():
        print(f"Processing {eval_name}...")
        answers.extend(create_context(model, tokenizer, eval_benchmark, minimal_prompt))
        if len(answers) > 5:
            break

    pd.DataFrame(answers, columns=["context", "A", "B", "C", "D", "label"]).to_csv(
        "outputs/hello_swag_output.csv", index=False
    )
