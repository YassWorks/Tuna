from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from helpers.utils import TextGenerator
import torch

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./output_lora"

text_gen = TextGenerator(MODEL_NAME)
tokenizer = text_gen.tokenizer
model = text_gen.model

# Load better dataset for PEFT
dataset = load_dataset("gsm8k", "main", split="test")
dataset = dataset.select(range(10))

# Preprocess function
def preprocess(example):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.squeeze() for k, v in encoded.items()}

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Define LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out = True
)

# Wrap model with PEFT
new_model = get_peft_model(model, peft_config)
new_model.print_trainable_parameters()

# Training setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=new_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# Merge LoRA weights into base model
print("#"*50)
print(model)
print("#"*50)
merged_model = new_model.merge_and_unload()
print("#"*50)
print(merged_model)
print("#"*50)

# Save the merged model and tokenizer
merged_model.save_pretrained(f"{OUTPUT_DIR}_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_merged")