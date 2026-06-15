import torch
from transformers import (
    # SmolLM2 is a causal language model (predicts next token based on previous tokens)
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    # Data collator to dynamically pad sequences in each batch (CPU optimization)
    DataCollatorForLanguageModeling
)
from config import Config
from data import get_tokenized_dataset

def setup_torch(config: Config):
    """Sets the number of threads for PyTorch to use, based on the config.
        This is a CPU optimization to prevent freezing during training."""
    torch.set_num_threads(config.torch_threads)

def load_model(config: Config):
    """Loads the pre-trained model from Hugging Face Transformers library"""
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float32,  #float32 for CPU training (CPU optimization)
    )
    return model

def get_training_args(config: Config):
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        # evaluate after every epoch
        eval_strategy="epoch",  
        # save checkpoint every epoch
        save_strategy="epoch",   
        # if training overfits, load the best model at the end 
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=False,    
        use_cpu=True # just CPU training, no GPU
    )

def train():
    """Main training function that orchestrates the entire training process:
    1. Loads the configuration
    2. Sets up PyTorch for CPU optimization
    3. Loads and tokenizes the dataset
    4. Loads the pre-trained model
    5. Sets up the Trainer and starts training
    6. Saves the fine-tuned model and pushes it to Hugging Face Hub"""
    config = Config()
    setup_torch(config)

    print("Loading data...")
    dataset, tokenizer = get_tokenized_dataset(config)

    print("Loading model...")
    model = load_model(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  #doing causal language modeling
        # padding automatically dynamically per batch to save memory (CPU optimization)
    )

    trainer = Trainer(
        model=model,
        args=get_training_args(config),
        train_dataset=dataset[config.train_split],
        eval_dataset=dataset[config.val_split],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("Pushing to HuggingFace Hub...")
    model.push_to_hub(config.hub_model_name)
    tokenizer.push_to_hub(config.hub_model_name)

if __name__ == "__main__":
    train()