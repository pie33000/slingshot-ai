# Slingshot AI

## Methodology

Let's define what perplexity is and how to interpret this measure. Perplexity is used to evaluate large language models by quantifying their uncertainty. The closer the metric is to 1, the more confident the model is in its predictions. Conversely, higher values indicate greater uncertainty in predicting the next token based on a given sequence.

Mathematically, perplexity is defined by the following formula:

- $X = (x_1, x_2, \ldots, x_t)$
- PPL(X) = exp( -1 / t * Σ (log(p_θ(x_i | x_<i))) )

Here, $X$ is a sequence of tokens, and `p_θ(x_i | x_<i)` is the probability of token $x_i$ given the preceding tokens.

---

In my methodology to evaluate perplexity and test efficiency on a Macbook Pro M1 Max, I made several approximations to compute perplexity on each counselor's messages:

- I always keep the system prompt. This allows me to evaluate the impact of different system prompts on perplexity.
- I limit the evaluation to a maximum of two chats, defined by a user request and a counselor answer. This approach aligns with evaluating models on long contexts as described in this [article](https://huggingface.co/docs/transformers/en/perplexity). When the limit is reached, the oldest chat is removed.
- Perplexity is only evaluated on the counselor's messages (i.e., the model's answers).
- To compute perplexity, I use Negative Log Likelihood (NLL). For a decoder-only model, NLL is equivalent to perplexity without the exponential term.

In addition to perplexity, I included a custom benchmark inspired by the [HellaSwag benchmark](https://arxiv.org/pdf/1905.07830). I created a dataset with a context (system prompt + user request) and generated three random answers using different LLMs, along with the real answer from the dataset.

The objective of this evaluation is to ask the model to choose the best possible ending based on the context from the four options. I am aware that since I am using open-source models, they are not fine-tuned, so the relevance of this method is limited. However, for a fine-tuned model, this benchmark provides a useful way to assess performance post-fine-tuning, in addition to using perplexity.

## Results

**Prompt A**

```
Your name is Ash. You are a counselor on a phone call.
Rules:
- You always speak in a professional manner.
- You always use perfect grammar and spelling.
- You vary the length and structure of your responses to sound natural.
```

**Prompt B**
```
Your name is Emmanuel Macron. You are the president of France.
Rules:
- You always speak formaly.
- You talk only about politics.
- You don't listen to people.
```
**⚠️The prompt is supposed to increase the perplexity of the model.**

| Model                           | Eval Dataset                | Prompt | PPL   |
|---------------------------------|-----------------------------|--------|-------|
| google/gemma-2b-it              | CAREER_DISSATISFACTION      | A      | 33.88 |
| google/gemma-2b-it              | CAREER_DISSATISFACTION      | B      | 57.89 |
| Qwen/Qwen2-1.5B-Instruct        | CAREER_DISSATISFACTION      | A      | 12.04 |
| Qwen/Qwen2-1.5B-Instruct        | CAREER_DISSATISFACTION      | B      | 12.58 |
| google/gemma-2b-it              | STUDY_ABROAD_DECISION       | A      | 40.58 |
| google/gemma-2b-it              | STUDY_ABROAD_DECISION       | B      | 48.03 |
| Qwen/Qwen2-1.5B-Instruct        | STUDY_ABROAD_DECISION       | A      | 13.48 |
| Qwen/Qwen2-1.5B-Instruct        | STUDY_ABROAD_DECISION       | B      | 13.37 |
| google/gemma-2b-it              | EMBRACING_A_HEALTHY_LIFESTYLE | A    | 27.48 |
| google/gemma-2b-it              | EMBRACING_A_HEALTHY_LIFESTYLE | B    | 37.43 |
| Qwen/Qwen2-1.5B-Instruct        | EMBRACING_A_HEALTHY_LIFESTYLE | A    | 10.87 |
| Qwen/Qwen2-1.5B-Instruct        | EMBRACING_A_HEALTHY_LIFESTYLE | B    | 11.15 |
| google/gemma-2b-it              | PARENTAL_OVERPROTECTEDNESS  | A      | 51.88 |
| google/gemma-2b-it              | PARENTAL_OVERPROTECTEDNESS  | B      | 70.62 |
| Qwen/Qwen2-1.5B-Instruct        | PARENTAL_OVERPROTECTEDNESS  | A      | 18.79 |
| Qwen/Qwen2-1.5B-Instruct        | PARENTAL_OVERPROTECTEDNESS  | B      | 19.99 |
| google/gemma-2b-it              | HANDLING_SETBACKS           | A      | 20.18 |
| google/gemma-2b-it              | HANDLING_SETBACKS           | B      | 22.82 |
| Qwen/Qwen2-1.5B-Instruct        | HANDLING_SETBACKS           | A      | 8.20  |
| Qwen/Qwen2-1.5B-Instruct        | HANDLING_SETBACKS           | B      | 8.30  |


|            Model             |       Eval Dataset       | Prompt | HelloSwag |
|:-----------------------------|:-------------------------|:------:|----------:|
| Qwen/Qwen2-1.5B-Instruct     | CAREER_DISSATISFACTION   |   A    |      0.27 |


## How to run the code

**Env set up**

    conda create -n slingshot python=3.11
    conda activate slingshot
    pip install -r requirements.txt

**Run perplexity evaluation**

    python calculate_ppl.py

The result will be saved in the output.txt file with the following format.
    {Eval Benchmark Name} {Prompt} {Perplexity}
    #Example
    HANDLING_SETBACKS Minimal System Prompt Perplexity: 20.1833885881935

**Run Custom Hello Swag Evaluation**

    python calculate_hello_swag.py