from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import secrets


class DataSet:
    """
    Wrapper for Hugging Face datasets.
    """
    def __init__(self, name: str, split: str = "train"):
        self.name = name
        self.split = split

    def load(self) -> Dataset:
        try:
            dataset: DatasetDict = load_dataset(self.name)
            if self.split not in dataset:
                raise ValueError(f"Split '{self.split}' not found in dataset '{self.name}'. Available: {list(dataset.keys())}")
            return dataset[self.split]
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{self.name}' with split '{self.split}': {e}")


class TextGenerator:
    """
    Wrapper for Hugging Face's AutoModelForCausalLM and AutoTokenizer.
    """
    def __init__(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Could not load model/tokenizer for '{model_name}': {e}")

        # Pad token handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def display(self):
        print("Model architecture " + "-"*30 + '\n', self.model, '\n' + "-"*30)


def random_hash(length: int = 8) -> str:
    """
    Generates a short random alphanumeric hash.
    """
    return secrets.token_hex(length // 2)
