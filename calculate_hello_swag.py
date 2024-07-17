import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(filepath: str) -> list[tuple[str]]:
    df = pd.read_csv(filepath)
    return df.to_records(index=False)


def process_data(tokenizer, data):
    context = data[0]
    endings = data[1], data[2], data[3], data[4]
    label = data[5]

    context_ids = tokenizer(context, return_tensors="pt").input_ids
    tok_rows, mask_rows = [], []

    for end in endings:
        end_tokens = tokenizer(" " + end, return_tensors="pt")
        tok_rows.append(torch.concat((context_ids[0], end_tokens.input_ids[0])))
        mask_rows.append([0] * len(context_ids[0]) + [1] * len(end_tokens.input_ids[0]))
    max_len = max(row.shape[0] for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


@torch.no_grad()
def evaluate(model, tokenizer, device):
    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in load_data("outputs/hello_swag_output.csv"):
        tokens, mask, label = process_data(tokenizer, example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens, attention_mask=mask).logits

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (
            mask[..., 1:]
        ).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"pred: {pred_norm}, label: {label}")
    print(
        f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
    )


if __name__ == "__main__":
    device = "mps"
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    evaluate(model, tokenizer, device)
