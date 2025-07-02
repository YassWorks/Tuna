from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tools.helpers.utils import TextGenerator, DataSet, random_hash
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
from tools.base import BaseTrainer, DEFAULT_OUTPUT_DIR
from typing import Any
import logging
import torch
import time
import os


class SFTTrainer(BaseTrainer):
    """
    Fine-tunes a given model using the Supervised Fine-Tuning (SFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """
    
    
    def __init__(
        self,
        text_gen: TextGenerator = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        test_dataset: DataSet | HFDataset | Any = None,
        test_dataset_name: str = None,
        test_dataset_split_name: str = "test",
        logger: logging.Logger = None
    ):
        """
        Initializes the SFTTrainer with either a TextGenerator instance or a model name, 
        and either a DataSet instance or a dataset name.
        Args:
            text_gen (TextGenerator, optional): An instance of TextGenerator. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            train_dataset (DataSet, optional): An instance of DataSet for training. Defaults to None.
            train_dataset_name (str, optional): The name of the training dataset to use. Defaults to None.
            train_dataset_split_name (str, optional): The split of the training dataset to use. Defaults to "train".
            test_dataset (DataSet, optional): An instance of DataSet for testing. Defaults to None.
            test_dataset_name (str, optional): The name of the testing dataset to use. Defaults to None.
            test_dataset_split_name (str, optional): The split of the testing dataset to use. Defaults to "test".
            logger (logging.Logger, optional): A logger instance for logging. Defaults to None.
        """
        
        # Initialize training dataset using BaseTrainer
        super().__init__(
            text_gen=text_gen,
            model_name=model_name,
            dataset=train_dataset,
            dataset_name=train_dataset_name,
            dataset_split_name=train_dataset_split_name,
            logger=logger
        )
        
        self.train_dataset = self.dataset
        
        # Configuring the Test Dataset
        if test_dataset_name:
            try:
                _dataset = DataSet(test_dataset_name, test_dataset_split_name)
                self.test_dataset = _dataset.load()
            except Exception:
                err_msg = "Please provide a valid test dataset split."
                self.logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif test_dataset:
            if isinstance(test_dataset, DataSet):
                try:
                    self.test_dataset = test_dataset.load()
                except Exception:
                    err_msg = "Please provide a valid test dataset split."
                    self.logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(test_dataset, HFDataset):
                self.test_dataset = test_dataset
        
        else:
            err_msg = "Either test_dataset_name or test_dataset must be provided"
            self.logger.exception(err_msg)
            raise ValueError(err_msg)


    def fine_tune(self, training_args: dict, save_to_disk: bool = False, output_dir: str = None, train_limit: int = None, test_limit: int = None, inplace: bool = False):
        """
        Fine-tunes the model using the provided dataset. (Supervised Fine-Tuning **SFT** approach)
        #### Note:
        This method will change the model's parameters **inplace** if inplace=True.
        Args:
            training_args (dict): A dictionary containing training arguments such as batch size, learning rate, etc.
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            train_limit (int, optional): Limit the number of training samples. Defaults to None.
            test_limit (int, optional): Limit the number of test samples. Defaults to None.
            inplace (bool, optional): Whether to fine-tune the model in-place or return a separate instance. Defaults to False.
        """
        
        try:
            if save_to_disk:
                if output_dir is None:
                    _hash = random_hash()
                    output_dir = DEFAULT_OUTPUT_DIR + f"/sft/sft_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"save_to_disk was set to 'True'. Output directory set to {output_dir}.")
                
                args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=training_args.get("per_device_train_batch_size", 4),
                    per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 4),
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-5),
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    logging_dir=f"{output_dir}/logs",
                    report_to="none",
                )
            else:
                # in-memory training without disk I/O
                args = TrainingArguments(
                    output_dir=DEFAULT_OUTPUT_DIR,  # required but won't be used
                    per_device_train_batch_size=training_args.get("per_device_train_batch_size", 4),
                    per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 4),
                    eval_strategy="epoch",
                    save_strategy="no",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-5),
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    report_to="none",
                    dataloader_pin_memory=False,
                    remove_unused_columns=True,
                )
            self.logger.info("Training arguments set up.")
        except Exception as e:
            err_msg = f"Failed to set up training arguments: {str(e)}"
            self.logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        tokenized_train = self.tokenize_dataset(self.train_dataset, train_limit)
        
        tokenized_test = self.tokenize_dataset(self.test_dataset, test_limit)
        
        if inplace:
            return super().start_fine_tune(
                tokenized_train=tokenized_train,
                tokenized_test=tokenized_test,
                training_args=args,
                inplace=inplace
            )
        else:
            return None
