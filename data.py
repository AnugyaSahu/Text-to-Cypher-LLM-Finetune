from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def load_data(config: Config):
    """Loads the dataset from Hugging Face Datasets library"""
    dataset = load_dataset(config.dataset_name)
    return dataset

def format_prompt(example):
    """Formats the example into the expected prompt format.
    The prompt includes the schema, question, and Cypher query in a structured format
    """
    return {
        "text": f"""### Schema:
                {example['schema']}

                ### Question:
                {example['question']}

                ### Cypher:
                {example['cypher']}"""
    }

def tokenize(example, tokenizer, config: Config):
    """
    Tokenizes the input text using the provided tokenizer.
    It truncates the text to the maximum length specified in the config (CPU optimization)
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

    dataset = load_data(config)
    dataset = dataset.map(format_prompt)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer, config))

    return dataset, tokenizer