from helpers.utils import evaluate_model_on_gsm8k, TextGenerator
from datasets import load_dataset

model = TextGenerator("distilgpt2")

# Load the GSM8K dataset (benchmark method)
gsm8k_dataset = load_dataset("gsm8k", "main", split="test")

acc = evaluate_model_on_gsm8k(model, gsm8k_dataset, limit=10)
print(acc)