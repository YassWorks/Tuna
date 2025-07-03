from backend.tools.helpers.utils import Model, DataSet, random_hash
from backend.tools.base import BaseTrainer, DEFAULT_OUTPUT_DIR
from transformers import TrainingArguments
from datasets import Dataset as HFDataset
from typing import Any
import logging
import torch
import os


class SFTTrainer(BaseTrainer):
    """
    Fine-tunes a given model using the Supervised Fine-Tuning (SFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """


    def __init__(
        self,
        text_gen: Model = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        test_dataset: DataSet | HFDataset | Any = None,
        test_dataset_name: str = None,
        test_dataset_split_name: str = "test",
        logger: logging.Logger = None,
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
            model=text_gen,
            model_name=model_name,
            train_dataset=train_dataset,
            train_dataset_name=train_dataset_name,
            train_dataset_split_name=train_dataset_split_name,
            evaluation=True,
            test_dataset=test_dataset,
            test_dataset_name=test_dataset_name,
            test_dataset_split_name=test_dataset_split_name,
            logger=logger,
        )


    def fine_tune(
        self,
        training_args: dict,
        save_to_disk: bool = False,
        output_dir: str = None,
        columns_train: list[str] = None,
        columns_test: list[str] = None,
        limit_train: int = None,
        limit_test: int = None,
        inplace: bool = False,
    ):
        """
        Fine-tunes the model using the provided dataset. (Supervised Fine-Tuning **SFT** approach)
        #### Note:
        This method will change the model's parameters **inplace** if inplace=True.
        Args:
            training_args (dict): A dictionary containing training arguments such as batch size, learning rate, etc.
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            train_limit (int, optional): Limit the number of training samples. Defaults to None.
            test_limit (int, optional): Limit the number of testing samples. Defaults to None.
            inplace (bool): If True, modifies the current instance in place. If False, returns a new instance. Defaults to False.
        Training Arguments:
            - `per_device_train_batch_size`: Batch size per device during training. Defaults to 4.
            - `num_train_epochs`: Number of epochs to train the model. Defaults to 3.
            - `learning_rate`: Learning rate for the optimizer. Defaults to 5e-4.
            - `warmup_steps`: Number of warmup steps for learning rate scheduler. Defaults to 100.
            - `weight_decay`: Weight decay for the optimizer. Defaults to 0.01.
        Returns:
            Model: The wrapper for the model (fine-tuned now) + tokenizer instance regardless of `inplace`.
        """

        try:
            if save_to_disk:

                if output_dir is None:
                    _hash = random_hash()
                    output_dir = DEFAULT_OUTPUT_DIR + f"/sft/sft_session_{_hash}"
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
                    per_device_eval_batch_size=training_args.get(
                        "per_device_eval_batch_size", 4
                    ),
                    eval_strategy="epoch",
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

                # in-memory training without disk I/O
                args = TrainingArguments(
                    output_dir=DEFAULT_OUTPUT_DIR,  # required but won't be used
                    per_device_train_batch_size=training_args.get(
                        "per_device_train_batch_size", 4
                    ),
                    per_device_eval_batch_size=training_args.get(
                        "per_device_eval_batch_size", 4
                    ),
                    eval_strategy="epoch",
                    save_strategy="no",
                    num_train_epochs=training_args.get("num_train_epochs", 3),
                    learning_rate=training_args.get("learning_rate", 5e-5),
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

        return super().start_fine_tune(
            training_args=args,
            inplace=inplace,
            columns_train=columns_train,
            columns_test=columns_test,
            limit_train=limit_train,
            limit_test=limit_test,
        )
