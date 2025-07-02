from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator, DataSet, random_hash
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from base import BaseTrainer, DEFAULT_OUTPUT_DIR
from typing import Any
import logging
import torch
import time
import os


class LoRATrainer(BaseTrainer):
    """
    Fine-tunes a given model using the Supervised Fine-Tuning (SFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory, or simple the new adapter layers separately.
    """
    def __init__(
        self,
        text_gen: TextGenerator = None,
        model_name: str = None,
        dataset: DataSet | HFDataset | Any = None,
        dataset_name: str = None,
        dataset_split_name: str = "train",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: int = 0.05,
        num_train_epochs: int = 3
    ):
        """
        Initializes the DAFTTrainer with either a TextGenerator instance or a model name, 
        and either a DataSet instance or a dataset name.
        Args:
            text_gen (TextGenerator, optional): An instance of TextGenerator. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            dataset (DataSet, optional): An instance of DataSet. Defaults to None.
            dataset_name (str, optional): The name of the dataset to use. Defaults to None.
            dataset_split_name (str, optional): The split of the dataset to use. Defaults to "train".
            lora_r (int, optional): LoRA rank. Defaults to 8.
            lora_alpha (int, optional): LoRA alpha. Defaults to 32.
            lora_dropout (float, optional): LoRA dropout rate. Defaults to 0.05.
            num_train_epochs (int, optional): Number of training epochs. Defaults to 3.
        """
        # Configuring the TextGenerator (model + tokenizer)
        if model_name:
            try:
                self.text_gen = TextGenerator(model_name)
            except Exception as e:
                err_msg = f"Failed to initialize TextGenerator with model_name '{model_name}': {str(e)}"
                logger.error(err_msg, exc_info=True)
                raise ValueError(err_msg)
            
        elif text_gen:
            if not isinstance(text_gen, TextGenerator):
                err_msg = "text_gen must be an instance of TextGenerator"
                logger.exception(err_msg)
                raise ValueError(err_msg)
            self.text_gen = text_gen
            
        else:
            err_msg = "Either model_name or text_gen must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)
        
        # Configuring the Dataset
        if dataset_name:
            try:
                _dataset = DataSet(dataset_name, dataset_split_name)
                self.dataset = _dataset.load()
            except Exception:
                err_msg = "Please provide a valid dataset split."
                logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif dataset:
            if isinstance(dataset, DataSet):
                try:
                    self.dataset = dataset.load()
                except Exception:
                    err_msg = "Please provide a valid dataset split."
                    logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(dataset, HFDataset):
                self.dataset = dataset # assuming it's good to go :3
        
        else:
            err_msg = "Either dataset_name or dataset must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)
        
        if not (1 <= lora_r <= 128):
            warn_msg = f"Unusual LoRA rank value: {lora_r}. It should be between 1 and 128. Setting it to 8."
            logger.warning(warn_msg)
            lora_r = 8
        else:
            self.lora_r = lora_r
        
        if not (1 <= lora_alpha <= 256):
            warn_msg = f"Unusual LoRA alpha value: {lora_alpha}. It should be between 1 and 256. Setting it to 32."
            logger.warning(warn_msg)
            lora_alpha = 32
        else:
            self.lora_alpha = lora_alpha

        if not (0.0 <= lora_dropout <= 0.5):
            warn_msg = f"Unusual LoRA dropout value: {lora_dropout}. It should be between 0.0 and 0.5. Setting it to 0.05."
            logger.warning(warn_msg)
            lora_dropout = 0.05
        else:
            self.lora_dropout = lora_dropout

        if not (1 <= num_train_epochs <= 100):
            warn_msg = f"Unusual number of training epochs: {num_train_epochs}. It should be between 1 and 100. Setting it to 3."
            logger.warning(warn_msg)
            num_train_epochs = 3
        else:
            self.num_train_epochs = num_train_epochs
        
    
    def fine_tune(self, save_to_disk: bool = False, output_dir: str = None, limit: int = None):
        """
        Fine-tunes the model using the provided dataset. (Low-Rank Adaptation **LoRA** approach)
        #### Note:
        This method will change the model's parameters **inplace**.
        Args:
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            limit (int, optional): Limit the number of training samples. Defaults to None.
        """
        model = self.text_gen.model
        tokenizer = self.text_gen.tokenizer
        _dataset = self.dataset
        
        if limit:
            if limit <= 0 or limit > len(self.dataset):
                err_msg = f"Limit must be a positive integer less than or equal to the size of the dataset. Current size: {len(self.dataset)}"
                logger.exception(err_msg)
                raise ValueError(err_msg)
            logger.info(f"Limiting dataset to {limit} samples.")
            _dataset = self.dataset.select(range(limit))
        
        # Apply the preprocessor to the selected limit of the dataset
        try:
            tokenized_train = _dataset.map(
                self.preprocessor, remove_columns=_dataset.column_names
            )
            logger.info(f"Tokenized dataset.")
        except Exception as e:
            err_msg = f"Failed to tokenize dataset: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            logger.info(f"Data collator created.")
        except Exception as e:
            err_msg = f"Failed to create data collator: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            if save_to_disk:
                if output_dir is None:
                    _hash = random_hash()
                    output_dir = OUTPUT_DIR + f"/lora/lora_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                logger.info(f"save_to_disk was set to 'True'. Output directory set to {output_dir}.")
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=4,
                    num_train_epochs=3,
                    learning_rate=5e-4,
                    logging_dir=f"{output_dir}/logs",
                    logging_steps=20,
                    fp16=torch.cuda.is_available(),
                    save_strategy="epoch",
                    report_to="none"
                )
            else:
                # in-memory training without disk I/O
                training_args = TrainingArguments(
                    output_dir=OUTPUT_DIR,  # required but won't be used
                    per_device_train_batch_size=4,
                    num_train_epochs=3,
                    learning_rate=5e-4,
                    fp16=torch.cuda.is_available(),
                    save_strategy="no",
                    report_to="none",
                    dataloader_pin_memory=False,
                    remove_unused_columns=True
                )
            logger.info(f"Training arguments set up.")
        except Exception as e:
            err_msg = f"Failed to set up training arguments: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=["c_attn"],
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=True
            )
            new_model = get_peft_model(model, peft_config)
            logger.info("Model wrapped with PEFT configuration.")
        except Exception as e:
            err_msg = f"Failed to wrap model with PEFT configuration: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            trainer = Trainer(
                model=new_model,
                args=training_args,
                train_dataset=tokenized_train,
                data_collator=data_collator
            )
            logger.info(f"Trainer initialized.")
        except Exception as e:
            err_msg = f"Failed to initialize Trainer: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            logger.info(f"Trainer starting...")
            trainer.train()
            logger.info(f"Training completed.")
        except Exception as e:
            err_msg = f"Training failed: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        # Merging and unloading LoRA layers
        try:
            merged = new_model.merge_and_unload()
            logger.info(f"LoRA layers merged and unloaded.")
            self.text_gen.model = merged
        except Exception as e:
            err_msg = f"Failed to merge and unload LoRA layers: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
    
        
    def preprocessor(self, example):
        """
        Preprocess function for the dataset.
        """
        full_text = " ".join([str(value) for value in example.values()])
        encoded = self.text_gen.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return {k: v.squeeze() for k, v in encoded.items()}
    
    
    def save_model(self, output_dir: str = None):
        """
        Saves the model and tokenizer to the specified output directory.
        Args:
            output_dir (str): The directory where the model and tokenizer will be saved.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR + "/final_lora"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.text_gen.model.save_pretrained(output_dir)
        self.text_gen.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}.")