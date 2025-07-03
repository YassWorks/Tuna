from backend.tools.helpers.utils import Model, DataSet, random_hash
from backend.tools.base import BaseTrainer, DEFAULT_OUTPUT_DIR
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from typing import Any
import logging
import torch
import copy
import os


class LoRATrainer(BaseTrainer):
    """
    Fine-tunes a given model using the Low-Rank Adaptation (LoRA) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory, or just the new adapter layers separately.
    """


    def __init__(
        self,
        model: Model = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        evaluation = False,
        logger: logging.Logger = None,
    ):
        """
        Initializes the LoRATrainer with either a Model instance or a model name,
        and either a DataSet instance or a dataset name.
        Args:
            model (Model, optional): An instance of Model. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            train_dataset (DataSet, optional): An instance of DataSet. Defaults to None.
            train_dataset_name (str, optional): The name of the dataset to use. Defaults to None.
            train_dataset_split_name (str, optional): The split of the dataset to use. Defaults to "train".
            lora_r (int, optional): LoRA rank. Defaults to 8.
            lora_alpha (int, optional): LoRA alpha. Defaults to 32.
            lora_dropout (float, optional): LoRA dropout rate. Defaults to 0.05.
            logger (logging.Logger, optional): A logger instance for logging. Defaults to None.
        """

        super().__init__(
            model=model,
            model_name=model_name,
            train_dataset=train_dataset,
            train_dataset_name=train_dataset_name,
            train_dataset_split_name=train_dataset_split_name,
            evaluation=evaluation,
            logger=logger,
        )


    def fine_tune(
        self,
        training_args: dict,
        LoRA_args: dict = None,
        save_to_disk: bool = False,
        output_dir: str = None,
        limit: int = None,
        inplace: bool = False,
    ):
        """
        Fine-tunes the model using the provided dataset. (Low-Rank Adaptation **LoRA** approach)
        Args:
            training_args (dict): A dictionary containing training arguments such as batch size, learning rate, etc.
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            limit (int, optional): Limit the number of training samples. Defaults to None.
            inplace (bool): If True, modifies the current instance in place. If False, returns a new instance. Defaults to False.
        Training Arguments:
            - `per_device_train_batch_size`: Batch size per device during training. Defaults to 4.
            - `num_train_epochs`: Number of epochs to train the model. Defaults to 3.
            - `learning_rate`: Learning rate for the optimizer. Defaults to 5e-4.
            - `warmup_steps`: Number of warmup steps for learning rate scheduler. Defaults to 100.
            - `weight_decay`: Weight decay for the optimizer. Defaults to 0.01.
        LoRA Arguments:
            - `r`: LoRA rank. Defaults to 8.
            - `lora_alpha`: LoRA alpha. Defaults to 32.
            - `lora_dropout`: LoRA dropout rate. Defaults to 0.05.
            - `target_modules`: List of target modules for LoRA. Defaults to None.
            - `bias`: Bias type for LoRA. Defaults to "none".
        """

        try:
            if save_to_disk:

                if output_dir is None:
                    _hash = random_hash()
                    output_dir = DEFAULT_OUTPUT_DIR + f"/lora/lora_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                if self.logger is not None:
                    self.logger.info(
                        f"save_to_disk was set to 'True'. Output directory set to {output_dir}."
                    )

                args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=training_args.get(
                        "per_device_train_batch_size", 4
                    ),
                    save_strategy="epoch",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-4),
                    warmup_steps=training_args.get("warmup_steps", 100),
                    weight_decay=training_args.get("weight_decay", 0.01),
                    fp16=torch.cuda.is_available(),
                    logging_dir=f"{output_dir}/logs",
                    report_to="none",
                )
            else:

                # in-memory training without disk I/O
                args = TrainingArguments(
                    output_dir=DEFAULT_OUTPUT_DIR,  # required but won't get used
                    per_device_train_batch_size=training_args.get(
                        "per_device_train_batch_size", 4
                    ),
                    save_strategy="no",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-4),
                    warmup_steps=training_args.get("warmup_steps", 100),
                    weight_decay=training_args.get("weight_decay", 0.01),
                    fp16=torch.cuda.is_available(),
                    report_to="none",
                )

            if self.logger is not None:
                self.logger.info("Training arguments set up.")

        except Exception as e:
            err_msg = f"Failed to set up training arguments: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            actual_lora_args = LoraConfig(
                r=LoRA_args.get("r", 8),
                lora_alpha=LoRA_args.get("lora_alpha", 32),
                target_modules=LoRA_args.get("target_modules", None),
                lora_dropout=LoRA_args.get("lora_dropout", 0.05),
                bias=LoRA_args.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=True,
            )
        except Exception as e:
            err_msg = f"Failed to create LoRA configuration: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        return self._apply_lora_and_train(args, actual_lora_args, inplace, limit)


    def _apply_lora_and_train(
        self, training_args: TrainingArguments, LoRA_args: LoraConfig, inplace: bool, limit: int
    ):
        """
        Applies LoRA configuration to the model and starts training.
        """

        base_model = self.model.model

        if not inplace:
            try:
                model = copy.deepcopy(base_model)
                if self.logger is not None:
                    self.logger.info("Model cloned for non-inplace training.")
            except Exception as e:
                err_msg = f"Failed to clone model: {str(e)}"
                if self.logger is not None:
                    self.logger.error(err_msg, exc_info=True)
                raise ValueError(err_msg)
        else:
            model = base_model

        try:
            peft_model = get_peft_model(model, LoRA_args)
            if self.logger is not None:
                self.logger.info("Model wrapped with LoRA configuration.")
        except Exception as e:
            err_msg = f"Failed to wrap model with LoRA configuration: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        tokenized_train = self.tokenize_dataset(self.train_dataset, limit)

        try:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.model.tokenizer, mlm=False
            )
            if self.logger is not None:
                self.logger.info("Data collator created.")
        except Exception as e:
            err_msg = f"Failed to create data collator: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_train,
                data_collator=data_collator,
            )
            if self.logger is not None:
                self.logger.info("Trainer initialized.")
        except Exception as e:
            err_msg = f"Failed to initialize Trainer: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            if self.logger is not None:
                self.logger.info("Starting LoRA training...")
            trainer.train()
            if self.logger is not None:
                self.logger.info("LoRA training completed.")
        except Exception as e:
            err_msg = f"LoRA training failed: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            if inplace:

                merged_model = peft_model.merge_and_unload()
                if self.logger is not None:
                    self.logger.info("LoRA layers merged and unloaded.")

                self.model.model = merged_model
                if self.logger is not None:
                    self.logger.info(
                        "Model updated in place with LoRA fine-tuned weights."
                    )

        except Exception as e:
            err_msg = f"Failed to merge and unload LoRA layers: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        return self.model
