from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator
import torch

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./output_daft_math"

text_gen = TextGenerator(MODEL_NAME)
tokenizer = text_gen.tokenizer
model = text_gen.model

# Load in-domain math data (unlabeled)
math_data = load_dataset("HuggingFaceH4/MATH-500")
train_dataset = math_data["test"].select(range(100))

# Preprocessing
def preprocess(example):
    full_text = " ".join([str(value) for value in example.values()])
    encoded = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.squeeze() for k, v in encoded.items()}

tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)

# Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)