import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_evaluation, write_output_to_file


def calculate_perplexity(model, tokenizer, prompt, text):
    prompt_tokens = tokenizer(prompt + "\n" + text, return_tensors="pt")
    text_tokens = tokenizer(text, return_tensors="pt")
    input_ids = prompt_tokens.input_ids
    start_idx = input_ids.shape[1] - text_tokens.input_ids.shape[1]
    perplexity_sequence = []
    for i in tqdm(range(start_idx, input_ids.shape[1])):
        with torch.no_grad():
            target = input_ids.clone()
            target[:, : i - 1] = -100
            outputs = model(input_ids[:, :i], labels=target[:, :i])
            perplexity_sequence.append(outputs.loss.item())
    return np.exp(np.mean(perplexity_sequence, axis=0))


def calculate_conversation_perplexity(
    model, tokenizer, eval_benchmark: list[str], system_prompt: str
) -> float:
    perplexity = []
    prompt = []
    for message in eval_benchmark:
        if message.startswith("user"):
            prompt.append(message)
        if message.startswith("assistant"):
            ppl = calculate_perplexity(
                model, tokenizer, system_prompt + "\n" + " ".join(prompt), message
            )
            prompt.append(message)
            perplexity.append(ppl)
        # I kept only 4 messages in the prompt as context to avoid memory consumption
        # and speed up the running time since the Attention Mechanism as a quadratic complexity
        if len(prompt) == 4:
            prompt = prompt[2:]
    average_perplexity = np.mean(perplexity, axis=0)
    return average_perplexity


# Load sample conversations from the provided zip file

if __name__ == "__main__":
    output_filepath = "output_qwen.txt"

    model_name = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Minimal System Prompt
    evals = load_evaluation("eval")

    minimal_prompt = """
    Your name is Ash. You are a counselor on a phone call.
    Rules:
    - You always speak in a professional manner.
    - You always use perfect grammar and spelling.
    - You vary the length and structure of your responses to sound natural."""

    # Changes expected to increase perplexity
    modified_prompt_increase = """
    Your name is Emmanuel Macron. You are the president of France.
    Rules:
    - You always speak formaly.
    - You talk only about politics.
    - You don't listen to people."""

    for eval_name, eval_benchmark in tqdm(evals.items()):
        print(f"Evaluating {eval_name}...")
        print(f"Minimal System Prompt: {minimal_prompt}")
        perplexity_minimal = calculate_conversation_perplexity(
            model, tokenizer, eval_benchmark, minimal_prompt
        )
        print(f"Maximal System Prompt: {modified_prompt_increase}")
        perplexity_maximal = calculate_conversation_perplexity(
            model, tokenizer, eval_benchmark, modified_prompt_increase
        )
        minimal_perplexity_info = f"{eval_name.upper()} Minimal System Prompt Perplexity: {perplexity_minimal}"
        maximal_perplexity_info = f"{eval_name.upper()} Maximal System Prompt Perplexity: {perplexity_maximal}"
        print(minimal_perplexity_info)
        print(maximal_perplexity_info)

        write_output_to_file(output_filepath, minimal_perplexity_info)
        write_output_to_file(output_filepath, maximal_perplexity_info)
