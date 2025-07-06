from peft import get_peft_model, TaskType, PromptTuningConfig, PeftModel
from tuna.tools.helpers.utils import Model, DataSet, random_hash
from tuna.tools.base import BaseTrainer, DEFAULT_OUTPUT_DIR
from transformers import TrainingArguments
from datasets import Dataset as HFDataset
from typing import Any
import logging
import torch
import os


class PTTrainer(BaseTrainer):
    """
    Fine-tunes a given model using the P-tuning approach on a given dataset.
    P-tuning introduces trainable continuous prompts (soft prompts) while keeping the model parameters frozen.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """


    def __init__(
        self,
        model: Model = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        evaluation: bool = False,
        logger: logging.Logger = None,
    ):
        """
        Initializes the PTTrainer with either a Model instance or a model name,
        and either a DataSet instance or a dataset name.
        Args:
            model (Model, optional): An instance of Model. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            train_dataset (DataSet, optional): An instance of DataSet. Defaults to None.
            train_dataset_name (str, optional): The name of the dataset to use. Defaults to None.
            train_dataset_split_name (str, optional): The split of the dataset to use. Defaults to "train".
            evaluation (bool, optional): Whether to enable evaluation during training. Defaults to False.
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
        training_args: dict = None,
        num_virtual_tokens: int = None,
        columns_train: list[str] = None,
        save_checkpoints: bool = False,
        output_dir: str = None,
        limit: int = None,
    ):
        """
        Fine-tunes the model using the provided dataset. (P-tuning approach)
        Args:
            training_args (dict): A dictionary containing training arguments such as batch size, learning rate, etc.
            num_virtual_tokens (int, optional): Number of trainable prompt tokens. Defaults to 20.
            columns_train (list[str], optional): Columns to use for training. Defaults to None.
            save_checkpoints (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            limit (int, optional): Limit the number of training samples. Defaults to None.
        Training Arguments:
            - `per_device_train_batch_size`: Batch size per device during training. Defaults to 4.
            - `num_train_epochs`: Number of epochs to train the model. Defaults to 3.
            - `learning_rate`: Learning rate for the optimizer. Defaults to 1e-3.
            - `warmup_steps`: Number of warmup steps for learning rate scheduler. Defaults to 100.
            - `weight_decay`: Weight decay for the optimizer. Defaults to 0.01.
        Returns:
            Model: The wrapper for the model.
        """
        
        try:
            if save_checkpoints:
                if output_dir is None:
                    _hash = random_hash()
                    output_dir = DEFAULT_OUTPUT_DIR + f"/sessions/ptuning_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                if self.logger is not None:
                    self.logger.info(
                        f"save_checkpoints was set to 'True'. Output directory set to {output_dir}."
                    )

                args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=training_args.get(
                        "per_device_train_batch_size", 4
                    ),
                    save_strategy="epoch",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 1e-3),
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
                    learning_rate=training_args.get("learning_rate", 1e-3),
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
            actual_pt_args = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens if num_virtual_tokens is not None else 20,
            )
        except Exception as e:
            err_msg = f"Failed to create P-tuning configuration: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)

        try:
            peft_model = get_peft_model(self.model.model, actual_pt_args)
            if self.logger is not None:
                self.logger.info("Model wrapped with P-tuning configuration.")
        except Exception as e:
            err_msg = f"Failed to wrap model with P-tuning configuration: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        self.model.model = peft_model
        
        fine_tuned_peft_model = super().start_fine_tune(
            training_args=args,
            columns_train=columns_train,
            limit_train=limit,
        )
        
        return fine_tuned_peft_model
