from backend.tools.helpers.utils import Model, DataSet, random_hash
from backend.tools.base import BaseTrainer, DEFAULT_OUTPUT_DIR
from transformers import TrainingArguments
from datasets import Dataset as HFDataset
from typing import Any
import logging
import torch
import os


class DAFTTrainer(BaseTrainer):
    """
    Fine-tunes a given model using the Domain-Adaptive Fine-Tuning (DAFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """
    
    
    def __init__(
        self,
        text_gen: Model = None,
        model_name: str = None,
        dataset: DataSet | HFDataset | Any = None,
        dataset_name: str = None,
        dataset_split_name: str = "train",
        logger: logging.Logger = None
    ):
        """
        Initializes the DAFTTrainer with either a TextGenerator instance or a model name, 
        and either a DataSet instance or a dataset name.
        Args:
            text_gen (TextGenerator, optional): An instance of TextGenerator. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            dataset (DataSet, optional): An instance of DataSet. Defaults to None.
            dataset_name (str, optional): The name of the dataset to use. Defaults to None.
            dataset_split (str, optional): The split of the dataset to use. Defaults to "train".
            logger (logging.Logger, optional): A logger instance for logging. Defaults to None.
        """
        
        super().__init__(
            model=text_gen,
            model_name=model_name,
            train_dataset=dataset,
            train_dataset_name=dataset_name,
            train_dataset_split_name=dataset_split_name,
            logger=logger
        )


    def fine_tune(self, training_args: dict, save_to_disk: bool = False, output_dir: str = None, limit: int = None, inplace: bool = False):
        """
        Fine-tunes the model using the provided dataset. (Domain-Adaptive Fine-Tuning **DAFT** approach)
        Args:
            training_args (dict): A dictionary containing training arguments such as batch size, learning rate, etc.
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            limit (int, optional): Limit the number of training samples. Defaults to None.
            inplace (bool): If True, modifies the current instance in place. If False, returns a new instance. Defaults to False.
        Training Arguments:
            - `per_device_train_batch_size`: Batch size per device during training.
            - `num_train_epochs`: Number of epochs to train the model.
            - `learning_rate`: Learning rate for the optimizer.
            - `warmup_steps`: Number of warmup steps for learning rate scheduler.
            - `weight_decay`: Weight decay for the optimizer.
        """
        
        try:
            if save_to_disk:
                
                if output_dir is None:
                    _hash = random_hash()
                    output_dir = DEFAULT_OUTPUT_DIR + f"/sessions/daft_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                if self.logger is not None:
                    self.logger.info(f"Saving training outputs to disk at {output_dir}")
                
                args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=training_args.get("per_device_train_batch_size", 4),
                    save_strategy="epoch",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-5),
                    warmup_steps=training_args.get("warmup_steps", 100),
                    weight_decay=training_args.get("weight_decay", 0.01),
                    fp16=torch.cuda.is_available(),
                    logging_dir=f"{output_dir}/logs",
                    report_to="none",
                )
            else:
                
                args = TrainingArguments(
                    output_dir=DEFAULT_OUTPUT_DIR, # required but won't get used
                    per_device_train_batch_size=training_args.get("per_device_train_batch_size", 4),
                    save_strategy="no",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-5),
                    warmup_steps=training_args.get("warmup_steps", 100),
                    weight_decay=training_args.get("weight_decay", 0.01),
                    fp16=torch.cuda.is_available(),
                    report_to="none",
                )
            
            if self.logger is not None:
                self.logger.info("TrainingArguments configured.")
            
        except Exception as e:
            err_msg = f"Failed to configure training arguments: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg)
            raise ValueError(err_msg)
        
        return self.start_fine_tune(
            training_args=args,
            inplace=inplace,
            limit_train=limit,
        )
