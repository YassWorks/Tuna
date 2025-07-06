from tuna import *
from datasets import load_dataset

logger = get_logger()

text_gen = Model(model_name="distilgpt2")

dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

trainer = LoRATrainer(
    model=text_gen,
    train_dataset=dataset,
    logger=logger,
)

training_args = {
    "per_device_train_batch_size": 2,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
}

lora_args = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["c_attn"],
    "bias": "none",
}

new_text_gen = trainer.fine_tune(
    training_args=training_args,
    LoRA_args=lora_args,
    limit=10
)

print(new_text_gen)
print("VS".center(50, "#"))
print(text_gen)

# error does NOT happen here
print(text_gen.generate("What is the integral of x^2?"))
# and error happens here
print(new_text_gen.generate("What is the integral of x^2?"))

trainer.save_model()