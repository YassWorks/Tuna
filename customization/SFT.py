from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator, evaluate_model_on_gsm8k
import torch

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./output_sft_gsm8k"

text_gen = TextGenerator(MODEL_NAME)
tokenizer = text_gen.tokenizer
model = text_gen.model

# Load GSM8K dataset
dataset = load_dataset("gsm8k", "main")
train_dataset = dataset["train"].select(range(100))
test_dataset = dataset["test"].select(range(30))

# Testinf before fine-tuning
print("Testing model before fine-tuning "+'#'*50)
score_pre = evaluate_model_on_gsm8k(text_gen, test_dataset, limit=10)
print(f"Accuracy before fine-tuning: {score_pre:.2f}")

# Preprocess function for SFT
def preprocess(example):
    prompt = example["question"]
    answer = example["answer"]
    full_text = f"{prompt}\n{answer}"
    
    encoded = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Causal LM: labels = input_ids
    encoded["labels"] = encoded["input_ids"].clone()
    
    return {k: v.squeeze() for k, v in encoded.items()}

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)

# Data collator (no MLM for causal models)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=torch.cuda.is_available(),
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Testinf after fine-tuning
print("Testing model after fine-tuning "+'#'*50)
new_text_gen = TextGenerator(OUTPUT_DIR)
score_post = evaluate_model_on_gsm8k(new_text_gen, test_dataset, limit=10)
print(f"Accuracy after fine-tuning: {score_post:.2f}")