from dataclasses import dataclass
import torch

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
    max_length: int = 1024

    # M2 GPU or cuda
    torch_threads: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Output
    output_dir: str = "my-model"
    results_path: str = "results/predictions.json"