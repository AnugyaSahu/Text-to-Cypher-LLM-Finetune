# evaluate.py

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from data import load_data, format_prompt, generate_cypher
from metrics import compute_metrics


def evaluate():
    config = Config()
    torch.set_num_threads(config.torch_threads)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(config.hub_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.hub_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # inference mode, no need to compute gradients
    model.eval()                 

    print("Loading test data...")
    dataset = load_data(config)
    test_data = dataset[config.test_split]

    results = []
    total_em = 0
    total_f1 = 0

    for i, example in enumerate(test_data):
        print(f"Sample {i+1}/{len(test_data)}")

        prediction = generate_cypher(
            model, tokenizer,
            example["schema"],
            example["question"],
            config
        )

        metrics = compute_metrics(prediction, example["cypher"])
        total_em += metrics["exact_match"]
        total_f1 += metrics["token_f1"]

        # per sample result
        results.append({
            "question": example["question"],
            "schema": example["schema"],
            "ground_truth": example["cypher"],
            "prediction": prediction,
            "exact_match": metrics["exact_match"],
            "token_f1": metrics["token_f1"]
        })

    # aggregate scores
    n = len(test_data)
    summary = {
        "aggregate": {
            "exact_match": round(total_em / n, 4),
            "token_f1": round(total_f1 / n, 4),
            "total_samples": n
        },
        "samples": results
    }

    # write to JSON
    os.makedirs("results", exist_ok=True)
    with open(config.results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExact Match: {summary['aggregate']['exact_match']}")
    print(f"Token F1:    {summary['aggregate']['token_f1']}")
    print(f"Results saved to {config.results_path}")


if __name__ == "__main__":
    evaluate()