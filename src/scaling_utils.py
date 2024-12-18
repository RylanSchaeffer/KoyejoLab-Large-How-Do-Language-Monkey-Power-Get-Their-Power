from datasets import load_dataset
import numpy as np
import pandas as pd
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple


# tokenizer("Question: ")
MODEL_HF_PATH_TO_QUESTION_TOKENS = {
    "google/gemma-2-2b": [9413, 235292],
    "google/gemma-2-9b": [9413, 235292],
    "meta-llama/Meta-Llama-3-8B": [],
    "Qwen/Qwen2-0.5B": [14582, 25],
    "Qwen/Qwen2-1.5B": [14582, 25],
}

# tokenizer("Answer: ")
MODEL_HF_PATH_TO_ANSWER_TOKENS = {
    "google/gemma-2-2b": [1261, 235292],
    "google/gemma-2-9b": [1261, 235292],
    "meta-llama/Meta-Llama-3-8B": [],
    "Qwen/Qwen2-0.5B": [16141, 25],
    "Qwen/Qwen2-1.5B": [16141, 25],
}


class ManyShotLM(object):
    def __init__(self, model_hf_path: str):
        self.model_hf_path = model_hf_path
        self.model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
        }
        #
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_hf_path, trust_remote_code=True
        )

    def compute_tokens_log_probs(self, encoded_questions_and_answers) -> torch.Tensor:
        # Get log probabilities for all encoded questions and answer.
        with torch.no_grad():
            input_ids = encoded_questions_and_answers.input_ids
            labels = input_ids.clone()
            labels = labels[:, 1:]  # Remove first position
            output = self.model(**encoded_questions_and_answers)
            logits = output.logits
            logits = logits[:, :-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            # Gather log probs for the actual tokens in the sequence.
            # Shape: (sequence_length,)
            tokens_log_probs = (
                torch.gather(log_probs, 2, labels.unsqueeze(-1))
                .squeeze()
                .cpu()
                .float()
                .numpy()
            )
        return tokens_log_probs


class GoogleGemma2(ManyShotLM):
    def __init__(self, model_hf_path: str = "google/gemma-2-2b"):
        super().__init__(model_hf_path=model_hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_hf_path,
            **self.model_kwargs,
        )


class GoogleGemma2IT(ManyShotLM):
    def __init__(self, model_hf_path: str = "google/gemma-2-2b-it"):
        super().__init__(model_hf_path=model_hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_hf_path,
            **self.model_kwargs,
        )


def extract_log_probs_of_many_shot_icl_answers(
    model_hf_path: str,
    questions_and_answers_token_ids: List[int],
    token_log_probs: np.ndarray,
):
    question_token_ids: List[int] = MODEL_HF_PATH_TO_QUESTION_TOKENS[model_hf_path]
    answer_token_ids: List[int] = MODEL_HF_PATH_TO_ANSWER_TOKENS[model_hf_path]

    # Identify the slices of token IDs corresponding to the answers.
    answer_start_indices = []
    question_start_indices = []
    for i in range(len(questions_and_answers_token_ids)):
        if (
            questions_and_answers_token_ids[i : i + len(question_token_ids)]
            == question_token_ids
        ):
            question_start_indices.append(i + len(question_token_ids))
        if (
            questions_and_answers_token_ids[i : i + len(answer_token_ids)]
            == answer_token_ids
        ):
            answer_start_indices.append(i + len(answer_token_ids))
    # Add length of questions and answers to handle last answer.
    question_start_indices.append(len(questions_and_answers_token_ids))

    sequence_indices = list(range(len(questions_and_answers_token_ids)))

    # Slice log probabilities of the answer tokens.
    log_probs_dict = {
        "log_probs": [],
        "Num. Shots": [],
        "Seq Idx": [],
    }
    for num_shots, (start_token_idx, end_token_idx_exclusive) in enumerate(
        zip(answer_start_indices[:-1], question_start_indices[1:])
    ):
        token_log_probs_slice = token_log_probs[start_token_idx:end_token_idx_exclusive]
        log_probs_dict["log_probs"].extend(token_log_probs_slice.tolist())
        log_probs_dict["Num. Shots"].extend([num_shots] * len(token_log_probs_slice))
        log_probs_dict["Seq Idx"].extend(
            sequence_indices[start_token_idx:end_token_idx_exclusive]
        )
    log_probs_df = pd.DataFrame.from_dict(log_probs_dict)
    return log_probs_df


def prepare_many_shot_icl_dataset(
    dataset_name: str,
) -> Tuple[List[str], List[str]]:
    if dataset_name == "CommonsenseQA":
        ds = load_dataset("tau/commonsense_qa", split="validation")
        choices_list: List[str] = [
            "".join(
                [
                    f"{label}. {text}\n"
                    for label, text in zip(choice["label"], choice["text"])
                ]
            )
            for choice in ds["choices"]
        ]
        questions: List[str] = [
            f"Question: {question}\nChoices:\n{choices}"
            for (question, choices) in zip(ds["question"], choices_list)
        ]
        answers: List[str] = [
            f"Answer: {answer}. {choices['text'][ord(answer) - ord('A')]}."
            for choices, answer in zip(ds["choices"], ds["answerKey"])
        ]
    elif dataset_name == "CREAK":

        def clean_up_question(question: str) -> str:
            return question.split("Input: ")[-1].replace("\nOutput: \n\n", "")

        ds = load_dataset(
            "Lots-of-LoRAs/task403_creak_commonsense_inference", split="test"
        )
        questions: List[str] = [
            f"Question: {clean_up_question(question)}" for question in ds["input"]
        ]
        answers = [f"Answer: {answer[0]}" for answer in ds["output"]]
    elif dataset_name == "LogiQA":
        ds = load_dataset("EleutherAI/logiqa", split="test")
        choices_list: List[str] = [
            "".join(
                [
                    f"{chr(ord('a') + option_idx)}. {option}\n"
                    for option_idx, option in enumerate(options)
                ]
            )
            for options in ds["options"]
        ]
        questions: List[str] = [
            f"Question: {context}\n{query}\nChoices:\n{choices}"
            for context, query, choices in zip(
                ds["context"], ds["question"], choices_list
            )
        ]
        answers: List[str] = [
            f"Answer: {label}. {option[ord(label) - ord('a')]}"
            for option, label in zip(ds["options"], ds["label"])
        ]
    elif dataset_name == "TriviaQA":
        ds = load_dataset("mrqa-workshop/mrqa", split="validation")
        ds = ds.filter(lambda x: x["subset"] == "TriviaQA-web", num_proc=10)
        questions: List[str] = [
            f"Question: {context} {question}"
            for context, question in zip(ds["context"], ds["question"])
        ]
        # Arbitrarily take the first answer.
        answers: List[str] = [f"Answer: {answer[0]}" for answer in ds["answers"]]
    elif dataset_name == "TruthfulQA":
        ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        ds = ds.filter(lambda x: x["type"] == "Non-Adversarial", num_proc=10)
        questions: List[str] = [f"Question: {question}" for question in ds["question"]]
        answers: List[str] = [f"Answer: {answer}" for answer in ds["best_answer"]]
    elif dataset_name == "WinoGrande":
        # ds = load_dataset("winogrande/winogrande_qa", split="validation")
        raise NotImplementedError("WinoGrande")
    else:
        raise NotImplementedError

    return questions, answers


def prepare_pretraining_causal_language_modeling_dataset(
    dataset_hf_path: str,
    **kwargs,
) -> List[str]:
    print(f"Creating {dataset_hf_path} sequences...")
    if dataset_hf_path == "allenai/c4":
        sequences: List[str] = load_dataset(
            dataset_hf_path,
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            trust_remote_code=True,
        )["text"]
    elif dataset_hf_path == "EleutherAI/lambada_openai":
        sequences: List[str] = load_dataset(
            dataset_hf_path, split="test", trust_remote_code=True
        )["text"]
    elif dataset_hf_path == "HuggingFaceFW/fineweb":
        sequences: List[str] = load_dataset(
            dataset_hf_path, "sample-10BT", split="train", trust_remote_code=True
        )["text"]
    elif dataset_hf_path == "JeanKaddour/minipile":
        sequences: List[str] = load_dataset(
            dataset_hf_path, split="test", trust_remote_code=True
        )["text"]
    elif dataset_hf_path == "monology/pile-test-val":
        sequences: List[str] = load_dataset(
            dataset_hf_path, split="test", trust_remote_code=True
        )["text"]
    elif dataset_hf_path == "togethercomputer/RedPajama-Data-1T-Sample":
        sequences: List[str] = load_dataset(
            dataset_hf_path, split="train", trust_remote_code=True
        )["text"]
    elif dataset_hf_path == "Zyphra/Zyda-2":
        sequences = load_dataset("Zyphra/Zyda-2", name="sample-100BT", split="train")[
            "text"
        ]
    else:
        raise NotImplementedError

    # Deterministically shuffle sequences.
    random.seed(0)
    random.shuffle(sequences)
    print(f"Created {dataset_hf_path}. Number of sequences: {len(sequences)}.")
    return sequences
