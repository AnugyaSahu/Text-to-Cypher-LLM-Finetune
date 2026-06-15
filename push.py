
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

config = Config()
model = AutoModelForCausalLM.from_pretrained(config.output_dir)
tokenizer = AutoTokenizer.from_pretrained(config.output_dir)

model.push_to_hub(config.hub_model_name)
tokenizer.push_to_hub(config.hub_model_name)