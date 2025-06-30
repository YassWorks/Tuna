from transformers import AutoModelForCausalLM, AutoTokenizer
from random import randint
from datasets import load_dataset


class DataSet:
    def __init__(self, name, split):
        self.name = name
        self.split = split

    def load(self):
        try:
            dataset = load_dataset(self.name)
            return dataset[self.split]
        except Exception as e:
            raise ValueError(f"Error occured while loading the dataset': {e}")


class TextGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
    

def random_hash(length=8):
    return ''.join([chr(randint(97, 122)) if randint(0, 1) else str(randint(0, 9)) for _ in range(length)])