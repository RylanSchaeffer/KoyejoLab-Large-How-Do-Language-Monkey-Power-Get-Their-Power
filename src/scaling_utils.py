from datasets import load_dataset
from pyarrow.dataset import dataset
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional


def prepare_in_context_scaling_questions_answers(
    dataset_path: str,
    dataset_name: Optional[str] = None,
) -> Dict[str, List[str]]:
    if dataset_path == "EleutherAI/logiqa":
        ds = load_dataset(dataset_path)["test"]
        prompts: List[str] = [
            f"{context} {query}"
            for context, query in zip(ds["context"], ds["question"])
        ]
        answers: List[str] = [
            option[ord(label) - ord("a")]
            for option, label in zip(ds["options"], ds["label"])
        ]
    elif dataset_path == "mrqa-workshop/mrqa" and dataset_name == "TriviaQA-web":
        # ds = load_dataset(dataset_path)
        # ds = ds.filter(lambda x: x["subset"] == dataset_name, num_proc=10)
        # print()
        raise NotImplementedError
    elif dataset_path == "truthfulqa/truthful_qa" and dataset_name == "generation":
        ds = load_dataset(dataset_path, dataset_name)["validation"]
        ds = ds.filter(lambda x: x["type"] == "Non-Adversarial", num_proc=10)
        prompts: List[str] = ds["question"]
        answers: List[str] = ds["best_answer"]
    else:
        raise NotImplementedError

    assert len(prompts) == len(answers)
    indices = list(range(len(prompts)))

    prompts_and_answers_dict = {
        "prompt": prompts,
        "answers": answers,
        "indices": indices,
    }
    return prompts_and_answers_dict


def prepare_pretraining_scaling_dataset(
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
