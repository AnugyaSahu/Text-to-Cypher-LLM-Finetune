# evaluate.py

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from data import load_data, format_prompt
from metrics import compute_metrics


def generate_cypher(model, tokenizer, schema: str, question: str, config: Config) -> str:
    """Generates a Cypher query given the schema and question using the fine-tuned model.
    — model learned from this exact format, changing it would confuse it at inference time
    """
    example = {"schema": schema, "question": question, "cypher": ""}
    prompt = format_prompt(example)["text"]
    
    # remove the empty cypher part, we want model to generate it
    prompt = prompt.split("### Cypher:")[0] + "### Cypher:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length
    )
    # no gradient needed at inference, saves memory
    with torch.no_grad():        
        outputs = model.generate(
            **inputs,
            # cypher queries are short
            max_new_tokens=128,   
            # deterministic output, reproducible results
            do_sample=False,
            eos_token_id = tokenizer.eos_token_id
        )

    # decode only the newly generated tokens, not the prompt
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    if "### Cypher:" in prediction:
        prediction = prediction.split("### Cypher:")[0].strip()

    return prediction


def evaluate():
    config = Config()
    torch.set_num_threads(config.torch_threads)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(config.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
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