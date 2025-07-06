# Tuna 🐟
### Fine-tuning made ridiculously easy

Stop wrestling with transformers boilerplate. Tuna gives you clean, powerful fine-tuning in just a few lines of code.

## Why Tuna?

**Before Tuna** (the painful way):
```python
# 50+ lines of boilerplate just to get started
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load and preprocess dataset
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
def preprocess_function(examples):
    # ... tokenization logic ...
    pass
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn"]
)
model = get_peft_model(model, peft_config)

# Set up training
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-4,
    # ... more args ...
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()
```

**With Tuna:**
```python
from tuna import *
from datasets import load_dataset

model = Model(model_name="distilgpt2")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

lora_trainer = LoRATrainer(model=model, train_dataset=dataset)
new_model = lora_trainer.fine_tune(
    training_args={"per_device_train_batch_size": 2, "num_train_epochs": 1},
    LoRA_args={"r": 8, "lora_alpha": 32, "target_modules": ["c_attn"]},
    limit=10
)
# That's it. Seriously.
```

## Features

### Multiple Fine-tuning Methods
- **LoRA** (Low-Rank Adaptation) - Memory efficient, fast
- **P-tuning** - Prompt tuning for few-shot learning
- **SFT** (Supervised Fine-Tuning) - Full parameter training with eval
- **DAFT** (Domain-Adaptive Fine-Tuning) - Specialized domain adaptation
- more coming soon...

### Flexible & Chainable
```python
# Chain different fine-tuning methods
model = Model("distilgpt2")
lora_model = LoRATrainer(model=model, train_dataset=dataset).fine_tune(...)
final_model = SFTTrainer(model=lora_model, train_dataset=train_dataset, eval_dataset=eval_dataset).fine_tune(...)
```

## Quick Start

```bash
pip install tuna # coming soon
```

```python
from tuna import *
from datasets import load_dataset

# Load your model and data
model = Model("microsoft/DialoGPT-small")
dataset = load_dataset("daily_dialog", split="train")

# Fine-tune with LoRA
trainer = LoRATrainer(model=model, train_dataset=dataset)
fine_tuned_model = trainer.fine_tune(
    training_args={
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
    },
    LoRA_args={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    },
    limit=1000  # Train on first 1000 samples
)

# Test your model
response = fine_tuned_model.generate("Hello, how are you?")
print(response)
```

## Advanced Usage

### Supervised Fine-Tuning with Evaluation
```python
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    test_dataset=test_data,  # Automatic evaluation
    logger=get_logger()
)

model = sft_trainer.fine_tune(
    training_args={"num_train_epochs": 5},
    save_checkpoints=True,  # Save progress
    columns_train=["text1", "text2"],  # Specify data columns
    limit_train=5000,
    limit_test=500
)
```

### Domain-Adaptive Fine-Tuning
```python
# Perfect for adapting models to specific domains
daft_trainer = DAFTTrainer(
    model_name="gpt2",
    dataset_name="medical_transcripts",
    logger=get_logger()
)

specialized_model = daft_trainer.fine_tune(
    training_args={"learning_rate": 2e-5},
    save_checkpoints=True,
    output_dir="./medical_gpt2"
)
```

### Prompt Tuning
```python
# Efficient few-shot learning
pt_trainer = PTTrainer(model=model, train_dataset=dataset)
prompt_tuned_model = pt_trainer.fine_tune(
    training_args={"num_train_epochs": 10},
    num_virtual_tokens=20,  # Number of learnable prompt tokens
    limit=100
)
```

## What Makes Tuna Special

**🎯 Zero Boilerplate** - No more copy-pasting training loops  
**🔄 Method Chaining** - Combine different fine-tuning approaches  
**📦 Smart Wrappers** - Handles model/tokenizer setup automatically  
**🎛️ Flexible Configuration** - Simple dicts for all parameters  
**💾 Session Management** - Automatic checkpointing and recovery  
**🔍 Built-in Logging** - Track everything without extra setup  

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT

---

PRs welcome!