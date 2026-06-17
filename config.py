from dataclasses import dataclass
import torch

# clean, no __init__ required
@dataclass
class Config:
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    # Finetuned model
    hub_model_name: str = "Anugya/text2cypher-smollm2"

    # Dataset
    dataset_name: str = "RomanTeucher/text2cypher-curated"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Training
    learning_rate: float = 2e-4 # standard
    num_epochs: int = 5 
    batch_size: int = 8 
    max_length: int = 1024 # some queries are large, no truncation

    # M2 GPU or cuda
    torch_threads: int = 8 # prevents cpu bottleneck on multi core machines
    # runs on cuda , mac or cpu without manual changes
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Output - locally save
    output_dir: str = "my-model"
    results_path: str = "results/predictions.json"