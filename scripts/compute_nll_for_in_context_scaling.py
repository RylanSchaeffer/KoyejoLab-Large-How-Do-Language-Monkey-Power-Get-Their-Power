import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict

import src.scaling_utils

model_names_to_max_context_lengths_dict = {
    "google/gemma-2-2b": 8192,
    "google/gemma-2-9b": 8192,
    "google/gemma-2-27b": 8192,
    "mistralai/Ministral-8B-Instruct-2410": 128_000,
    "Qwen/Qwen2-7B-Instruct": 32_768,
}

paths_and_optional_names_list = [
    {"dataset_path": "EleutherAI/logiqa"},
    {"dataset_path": "mrqa-workshop/mrqa", "dataset_name": "TriviaQA-web"},
    {"dataset_path": "truthfulqa/truthful_qa", "dataset_name": "generation"},
    {"dataset_path": "cais/mmlu", "dataset_name": "all"},
]

# for model_name in model_names_to_max_context_lengths_dict.keys():
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

for model_name in model_names_to_max_context_lengths_dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    for path_and_optional_name in paths_and_optional_names_list:
        prompts_and_answers_dict = (
            src.msj_utils.prepare_in_context_scaling_questions_answers(
                **path_and_optional_name,
            )
        )
        print()


# pkl_file_path = os.path.join(
#     "data",
#     "raw_data",
#     "many_shot_jailbreaking",
#     "task_scary_evals_psychopathy__max_num_shots_300__num_examples_301.pkl",
# )
# with open(pkl_file_path, "rb") as f:
#     samples_list = pickle.load(f)
#
# data = pd.read_json(
#     "data/raw_data/many_shot_jailbreaking/opportunity_to_insult.jsonl", lines=True
# )


print("Finished generating NLL for MSJ!")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set seaborn style to whitegrid.
sns.set_style("whitegrid")

force_refresh = False
# force_refresh = True


LLAMA_SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

speaker2role = {
    "Human": "user",
    "Assistant": "assistant",
}


def convert_sample_to_llama2_prompts(
    sample, delim=": ", speaker2role: Dict[str, str] = speaker2role
):
    turns = sample.lstrip("\n").split("\n\n")

    messages, answers = [], []
    for turn in turns:
        assert delim in turn, turn
        speaker, text = turn.split(delim, 1)
        assert speaker in speaker2role.keys()
        assert not turn.endswith("\n")
        if speaker == "Human":
            messages.append(f"{text} [/INST] ")
            answers.append(None)
        elif speaker == "Assistant":
            messages.append(f"{text} </s><s>[INST] ")
            answers.append(text)

    return messages


def convert_sample_to_mistral_prompts(sample, delim=": ", speaker2role=speaker2role):
    turns = sample.lstrip("\n").split("\n\n")
    messages = []
    for turn in turns:
        assert delim in turn, turn
        speaker, text = turn.split(delim, 1)
        assert speaker in speaker2role.keys()
        assert not turn.endswith("\n")
        messages.append(
            {
                "role": speaker2role[speaker],
                "content": text,
            }
        )
    return messages


results = []
os.makedirs(results_dir, exist_ok=True)
for model_name in model_names_to_max_context_lengths_dict.keys():
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    model_results_path = os.path.join(model_results_dir, "results.csv")
    if not force_refresh and os.path.exists(model_results_path):
        print(f"Skipping {model_name} because results already exist.")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # rope_scaling = {"type": "dynamic", "factor": scaling_factor}
    )

    # Lightly modify the tokenizer and get the token IDs.
    tokenizer.pad_token = tokenizer.eos_token
    if "mistralai" in model_name and model_name != "mistralai/Mistral-7B-v0.1":
        # TODO: Decide how to handle this.
        # For whatever reason, tokenizer.encode() encodes "Yes" as 5592 and "No" as 1770.
        # But tokenizer.apply_chat_template encodes "Yes" as 5613 and "No" as 2501.
        # Easiest solution is to override here.
        # Would a more robust solution be to encode the prompt, then decode it, and
        # then find the token IDs of "Yes" and "No"? That way we don't have to hardcode.
        yes_token_id = 5613
        no_token_id = 2501
    else:
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    if "meta-llama" in model_name:
        prompts_per_sample = [
            LLAMA_SYSTEM_PROMPT + "".join(convert_sample_to_llama2_prompts(sample))
            for sample in samples_list
        ]
    elif "mistralai" in model_name:
        prompts_per_sample = [
            tokenizer.apply_chat_template(
                convert_sample_to_mistral_prompts(sample=sample), tokenize=False
            )
            for sample in samples_list
        ]
    elif "togethercompute" in model_name:
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    model_message_results_df_list = []
    for sample_idx, prompt in tqdm(enumerate(prompts_per_sample)):
        print(f"Model: {model_name}\tSample: {sample_idx}\tTime: {time.time()}")

        # Encode the prompt using the tokenizer.
        prompt_encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            # padding="max_length",
            truncation=True,  # Downsize prompt to fit in the context window.
            max_length=model_names_to_max_context_lengths_dict[model_name],
        )

        # If you want to convert the input IDs back to string tokens:
        # tokens = tokenizer.convert_ids_to_tokens(prompt_encoded["input_ids"][0])
        tokenized_prompt_len = prompt_encoded["input_ids"].shape[1]

        # We shift by -1 because we want the probability mass the model places
        # when predicting the yes or no tokens.
        yes_token_seq_indices = (
            torch.where(prompt_encoded["input_ids"] == yes_token_id)[1] - 1
        )
        no_token_seq_indices = (
            torch.where(prompt_encoded["input_ids"] == no_token_id)[1] - 1
        )
        assert len(yes_token_seq_indices) > 0
        assert len(no_token_seq_indices) > 0
        assert (len(yes_token_seq_indices) + len(no_token_seq_indices)) <= 301

        # Generate a response from the model.
        input_ids = prompt_encoded["input_ids"].to(model.device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                labels=input_ids,
            )
            logits = output.logits

        yes_logits = logits[0, yes_token_seq_indices, yes_token_id]
        no_logits = logits[0, no_token_seq_indices, no_token_id]

        yes_probs = torch.softmax(logits[0, yes_token_seq_indices, :], dim=1)[
            :, yes_token_id
        ]
        no_probs = torch.softmax(logits[0, no_token_seq_indices, :], dim=1)[
            :, no_token_id
        ]
        seq_indices = (
            torch.cat([yes_token_seq_indices, no_token_seq_indices])
            .detach()
            .cpu()
            .numpy()
        )
        argsort_indices = np.argsort(seq_indices)
        logits_arr = (
            torch.cat([yes_logits, no_logits]).detach().cpu().numpy()[argsort_indices]
        )
        probs_arr = (
            torch.cat([yes_probs, no_probs]).detach().cpu().numpy()[argsort_indices]
        )
        nll_arr = -np.log(probs_arr)[argsort_indices]
        n_shot_arr = np.arange(len(argsort_indices))

        model_message_results_df = pd.DataFrame(
            {
                "n_shot": n_shot_arr,
                "logit": logits_arr,
                "prob": probs_arr,
                "nll": nll_arr,
            }
        )
        model_message_results_df["model_name"] = model_name
        model_message_results_df["message_no"] = sample_idx
        model_message_results_df_list.append(model_message_results_df)

    model_results_df = pd.concat(model_message_results_df_list)
    model_results_df.to_csv(model_results_path)
    print(f"{model_name} Shape: ", model_results_df.shape)
    print(model_results_df.head())

# Load data from disk.
model_results_dfs_list = []
for model_name in model_names_to_max_context_lengths_dict.keys():
    try:
        df = pd.read_csv(os.path.join(results_dir, model_name, "results.csv"))
        model_results_dfs_list.append(df)
    except FileNotFoundError:
        continue
results_df = pd.concat(model_results_dfs_list)

plt.close()
ax = sns.lineplot(results_df, x="n_shot", y="nll", markers="x", hue="model_name")
ax.set_xscale("log")
ax.set_yscale("log")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.0, 1.0))
plt.savefig(os.path.join(results_dir, "nll_scaling_fig.png"))
# plt.show()

plt.close()
ax = sns.lineplot(results_df, x="n_shot", y="prob", markers="x", hue="model_name")
ax.set_xscale("log")
ax.set_yscale("log")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.0, 1.0))
plt.savefig(os.path.join(results_dir, "prob_scaling_fig.png"))
# plt.show()
