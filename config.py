
from dataclasses import dataclass

# dataclass because cleaner than a dict and more structured, also allows for type hints and default values
@dataclass
class Config:
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    hub_model_name: str = "Anugya/text2cypher-smollm2"

    # Dataset
    dataset_name: str = "RomanTeucher/text2cypher-curated"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 5
    batch_size: int = 8
    max_length: int = 512   

    # M2 GPU
    torch_threads: int = 4
    device: str = "mps"        

    # Output
    output_dir: str = "./my-model"
    results_path: str = "./results/predictions.json"