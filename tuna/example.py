from tuna import *
from datasets import load_dataset

logger = get_logger()

old_model = Model(model_name="distilgpt2")

model = Model(model_name="distilgpt2")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

lora_trainer = LoRATrainer(
    model=model,
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

new_model = lora_trainer.fine_tune(
    training_args=training_args,
    LoRA_args=lora_args,
    limit=10
)

print(" OLD ".center(50, "#"))
print(old_model)
print(" NEW ".center(50, "#"))
print(new_model)

pt_trainer = PTTrainer(
    model=new_model,
    train_dataset=dataset,
    logger=logger,
)

new_new_model = pt_trainer.fine_tune(
    training_args=training_args,
    limit=10,
)

print(" NEW NEW ".center(50, "#"))
print(new_model)
