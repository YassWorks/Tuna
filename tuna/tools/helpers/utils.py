from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import secrets
import torch


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
            dataset: DatasetDict = load_dataset(name=self.name, split=self.split, **self.kwargs)
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
    def __init__(self, model = None, tokenizer = None, model_name: str = None, tokenizer_kwargs=None, model_kwargs=None):
        try:
            if model:
                self.model = model
            elif model_name:
                model_kwargs = model_kwargs or {}
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                raise ValueError("Either 'model' or 'model_name' must be provided.")
            
            if tokenizer:
                self.tokenizer = tokenizer
            elif model_name:
                tokenizer_kwargs = tokenizer_kwargs or {}
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            else:
                raise ValueError("Either 'tokenizer' or 'model_name' must be provided.")
            
        except Exception as e:
            raise ValueError(f"Could not load model/tokenizer for '{model_name}': {e}")

        # Pad token handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate(self, prompt: str, max_length: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.strip()
    
    def __str__(self):
        return " Model architecture ".center(100, '-') + '\n' + str(self.model) + '\n' + "-"*100
    
    def __repr__(self):
        return self.__str__()


def random_hash(length: int = 8) -> str:
    """
    Generates a short random alphanumeric hash.
    """
    return secrets.token_hex(length // 2)
