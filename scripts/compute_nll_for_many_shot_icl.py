import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple

import src.globals
import src.scaling_utils


raw_data_dir = "data/raw_data/many_shot_icl"
os.makedirs(raw_data_dir, exist_ok=True)

max_context_length = 2048
model_nicknames_to_huggingface_paths_dict = {
    # "Cerebras_111M_2B": "cerebras/Cerebras-GPT-111M",
    # "Cerebras_256M_5B": "cerebras/Cerebras-GPT-256M",
    # "Cerebras_590M_11B": "cerebras/Cerebras-GPT-590M",
    # "Cerebras_1.3B_26B": "cerebras/Cerebras-GPT-1.3B",
    # "Cerebras_2.7B_54B": "cerebras/Cerebras-GPT-2.7B",
    # "Cerebras_6.7B_134B": "cerebras/Cerebras-GPT-6.7B",
    # "Cerebras_13B_260B": "cerebras/Cerebras-GPT-13B",
    "Gemma 2 2B": ("google/gemma-2-2b", 8192),
    # "Gemma 2 9B": ("google/gemma-2-9b", 8192),
    # "Llama 3 8B": ("meta-llama/Meta-Llama-3-8B", 8192),
    # "Qwen 2.5 7B": ("Qwen/Qwen2.5-7B", 131072),
    # "Mistral v0.3 7B": ("mistralai/Mistral-7B-v0.3", 32768),
}

for dataset_name in src.globals.MANY_SHOT_IN_CONTEXT_LEARNING_DATASET_ORDER:
    (
        questions,
        answers,
    ) = src.scaling_utils.prepare_many_shot_icl_dataset(
        dataset_name=dataset_name,
    )

    for (
        model_nickname,
        (model_hf_path, max_context_length),
    ) in model_nicknames_to_huggingface_paths_dict.items():
        tokenizer = AutoTokenizer.from_pretrained(model_hf_path, trust_remote_code=True)

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
        }
        if "gemma" in model_hf_path:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_hf_path,
            **model_kwargs,
        )

        for shuffle_idx in range(15):
            # Permute questions and answers.
            random.seed(shuffle_idx)
            shuffled_data = list(zip(questions, answers))
            random.shuffle(shuffled_data)
            questions, answers = zip(*shuffled_data)

            questions_and_answers = "\n".join(
                [
                    f"{question}\n{answer}\n\n"
                    for question, answer in zip(questions, answers)
                ]
            )

            encoded_questions_and_answers = tokenizer(
                questions_and_answers,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_context_length,
            ).to(model.device)

            # Get log probabilities for all encoded questions and answer.
            with torch.no_grad():
                input_ids = encoded_questions_and_answers.input_ids
                labels = input_ids.clone()
                labels = labels[:, 1:]  # Remove first position
                output = model(**encoded_questions_and_answers)
                logits = output.logits
                logits = logits[:, :-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                # Gather log probs for the actual tokens in the sequence.
                # Shape: (sequence_length,)
                token_log_probs = (
                    torch.gather(log_probs, 2, labels.unsqueeze(-1))
                    .squeeze()
                    .cpu()
                    .float()
                    .numpy()
                )

            questions_and_answers_token_ids: List[int] = (
                encoded_questions_and_answers.input_ids[0].cpu().numpy().tolist()
            )

            log_probs_df = src.scaling_utils.extract_log_probs_of_many_shot_icl_answers(
                model_hf_path=model_hf_path,
                questions_and_answers_token_ids=questions_and_answers_token_ids,
                token_log_probs=token_log_probs,
            )

            log_probs_df["Model Nickname"] = model_nickname
            log_probs_df["Dataset"] = dataset_name
            log_probs_df["Shuffle Idx"] = shuffle_idx

            log_probs_df.to_parquet(
                os.path.join(
                    raw_data_dir,
                    f"{dataset_name}-{model_nickname}-shuffle={shuffle_idx}_log_probs.parquet",
                ),
                index=False,
            )
            print(
                f"Saved log probs for {model_nickname} on {dataset_name} for shuffle {shuffle_idx}."
            )

        del model, tokenizer, log_probs_df
