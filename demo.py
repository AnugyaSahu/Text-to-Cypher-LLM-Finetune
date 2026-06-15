import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from data import load_data, format_prompt

def generate_cypher(model, tokenizer, schema, question, config):
    example = {"schema": schema, "question": question, "cypher": ""}
    prompt = format_prompt(example)["text"]
    prompt = prompt.split("### Cypher:")[0] + "### Cypher:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def print_example(schema, question, ground_truth=None, prediction=None, index=None, label=""):
    print(f"\n{'='*60}")
    print(f"  {label} {index}")
    print(f"{'='*60}")
    print(f"  Schema:       {schema}")
    print(f"  Question:     {question}")
    if ground_truth:
        print(f"  Ground Truth: {ground_truth}")
    print(f"  Predicted:    {prediction}")


def run_demo():
    config = Config()
    torch.set_num_threads(config.torch_threads)

    print("Loading model from HuggingFace Hub...")
    model = AutoModelForCausalLM.from_pretrained(config.hub_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.hub_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # --- Test set examples ---
    print("\nLoading test data...")
    dataset = load_data(config)
    test_data = dataset[config.test_split]
    random_indices = random.sample(range(len(test_data)), 3)

    print("\n\n" + " "*20 + "FROM TEST SET (with Ground Truth)")
    for i, idx in enumerate(random_indices):
        example = test_data[idx]
        prediction = generate_cypher(
            model, tokenizer,
            example["schema"],
            example["question"],
            config
        )
        print_example(
            schema=example["schema"],
            question=example["question"],
            ground_truth=example["cypher"],
            prediction=prediction,
            index=i+1,
            label="Test Example"
        )

    # --- Custom made up examples ---
    custom_examples = [
        {
            "schema": "Movie {title, year}, Person {name}, (Person)-[:DIRECTED]->(Movie)",
            "question": "Which movies did Christopher Nolan direct before 2010?"
        },
        {
            "schema": "Product {name, price}, Category {name}, (Product)-[:BELONGS_TO]->(Category)",
            "question": "What products belong to the Electronics category?"
        }
    ]

    print("\n\n" + " "*20 + "CUSTOM EXAMPLES (Out of Dataset)")
    for i, ex in enumerate(custom_examples):
        prediction = generate_cypher(
            model, tokenizer,
            ex["schema"],
            ex["question"],
            config
        )
        print_example(
            schema=ex["schema"],
            question=ex["question"],
            prediction=prediction,
            index=i+1,
            label="Custom Example"
        )

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    run_demo()