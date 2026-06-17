from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def load_data(config: Config):
    """Loads the dataset from Hugging Face Datasets library"""
    dataset = load_dataset(config.dataset_name)
    return dataset

def format_prompt(example, eos_token=""):
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
        # Efficient to pad each batch during training (Dynamic padding)
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