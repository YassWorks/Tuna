from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import secrets


class DataSet:
    """
    Wrapper for Hugging Face datasets.
    """
    def __init__(self, name: str, split: str = "train", **kwargs):
        self.name = name
        self.split = split
        self.kwargs = kwargs

    def load(self) -> Dataset:
        try:
            dataset: DatasetDict = load_dataset(self.name, **self.kwargs)
            if self.split not in dataset:
                raise ValueError(f"Split '{self.split}' not found in dataset '{self.name}'. Available: {list(dataset.keys())}")
            return dataset[self.split]
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{self.name}' with split '{self.split}': {e}")
        
    def __str__(self):
        return f"DataSet(name={self.name}, split={self.split}, kwargs={self.kwargs})"
    
    def __repr__(self):
        return self.__str__()


class Model:
    """
    Wrapper for Hugging Face's AutoModelForCausalLM and AutoTokenizer.
    """
    def __init__(self, model_name: str, tokenizer_kwargs=None, model_kwargs=None):
        try:
            tokenizer_kwargs = tokenizer_kwargs or {}
            model_kwargs = model_kwargs or {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            raise ValueError(f"Could not load model/tokenizer for '{model_name}': {e}")

        # Pad token handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate(self, prompt: str, **gen_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    def __str__(self):
        return "Model architecture " + "-"*30 + '\n' + str(self.model) + '\n' + "-"*30
    
    def __repr__(self):
        return self.__str__()


def random_hash(length: int = 8) -> str:
    """
    Generates a short random alphanumeric hash.
    """
    return secrets.token_hex(length // 2)
