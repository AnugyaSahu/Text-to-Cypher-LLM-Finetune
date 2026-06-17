from datasets import load_dataset
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from config import Config

def load_data(config: Config):
    """Loads the dataset from Hugging Face Datasets library"""
    dataset = load_dataset(config.dataset_name)
    return dataset

def format_prompt(example, eos_token=""):
    # eos token so model learns when to stop, default empty so it works on inference too
    return {
        "text": f"""### Schema:
{example['schema']}
### Question:
{example['question']}
### Cypher:
{example['cypher']}{eos_token}"""
    }

def tokenize(example, tokenizer, config: Config):
    """
    Tokenizes the input text using the provided tokenizer.
    It truncates the text to the maximum length specified in the config
    """
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=config.max_length,
        # padding happens dynaically by batches in training
        padding=False,
    )

def get_tokenized_dataset(config: Config):
    """Loads the dataset, formats the prompts, and tokenizes the text.
    Returns the tokenized dataset and the tokenizer for later use in training and inference."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # SmolLM2 doesn't have a pad token by default
    # so we set it to eos_token to avoid errors (end of sequence token)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_data(config).map(lambda x: tokenize(format_prompt(x, tokenizer.eos_token), tokenizer, config))

    return dataset, tokenizer

class NewlineStoppingCriteria(StoppingCriteria):
    # cypher queries are single line, force stopping at new line 
    def __init__(self, tokenizer):
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] == self.newline_id

def generate_cypher(model, tokenizer, schema: str, question: str, config) -> str:
    example = {"schema": schema, "question": question, "cypher": ""}
    prompt = format_prompt(example)["text"]
    prompt = prompt.split("### Cypher:")[0] + "### Cypher:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length
    )

    # inference does not need gradients
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            # for deterministic outputs
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([NewlineStoppingCriteria(tokenizer)])
        )

    # model returns prompt+generated, slice and decode only newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return prediction