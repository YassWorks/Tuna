from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from backend.tools.helpers.utils import Model, DataSet, random_hash
from transformers import AutoModelForCausalLM
from datasets import Dataset as HFDataset
from typing import Any
import logging
import os


DEFAULT_OUTPUT_DIR = "./.cache/output"


class BaseTrainer:
    """Base class for Trainer classes."""
    
    
    def __init__(
        self,
        model: Model = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        evaluation: bool = False,
        test_dataset: DataSet | HFDataset | Any = None,
        test_dataset_name: str = None,
        test_dataset_split_name: str = "train",
        logger: logging.Logger = None,
    ):
        """
        Initializes the BaseTrainer with either a TextGenerator instance or a model name,
        and either a DataSet instance or a dataset name.
        Args:
            model (Model, optional): An instance of Model. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            train_dataset (DataSet | HFDataset | Any, optional): An instance of DataSet or HFDataset for training. Defaults to None.
            train_dataset_name (str, optional): The name of the training dataset. Defaults to None.
            train_dataset_split_name (str, optional): The split of the training dataset to use. Defaults to "train".
            evaluation (bool, optional): Whether to enable evaluation during training. Defaults to False.
            test_dataset (DataSet | HFDataset | Any, optional): An instance of DataSet or HFDataset for testing. Defaults to None.
            test_dataset_name (str, optional): The name of the testing dataset. Defaults to None.
            test_dataset_split_name (str, optional): The split of the testing dataset to use. Defaults to "train".
            logger (logging.Logger, optional): A logger instance for logging. Defaults to None.
        """
        
        self.logger = logger
        
        # Configuring the TextGenerator (model + tokenizer)
        if model_name:
            try:
                self.model = Model(model_name)
            except Exception as e:
                err_msg = f"Failed to initialize TextGenerator with model_name '{model_name}': {str(e)}"
                if self.logger is not None:
                    self.logger.error(err_msg, exc_info=True)
                raise ValueError(err_msg)
            
        elif model:
            if not isinstance(model, Model):
                err_msg = "text_gen must be an instance of TextGenerator"
                if self.logger is not None:
                    self.logger.exception(err_msg)
                raise ValueError(err_msg)
            self.model = model
            
        else:
            err_msg = "Either model_name or text_gen must be provided"
            if self.logger is not None:
                self.logger.exception(err_msg)
            raise ValueError(err_msg)
        
        self.train_dataset = self.configure_dataset(
            dataset=train_dataset,
            dataset_name=train_dataset_name,
            dataset_split_name=train_dataset_split_name
        )
        
        self.evaluation = evaluation
        if evaluation:
            self.test_dataset = self.configure_dataset(
                dataset=test_dataset,
                dataset_name=test_dataset_name,
                dataset_split_name=test_dataset_split_name
            )
    
    
    def configure_dataset(self, dataset: DataSet | HFDataset | Any = None, dataset_name: str = None, dataset_split_name: str = None) -> HFDataset:
        """Configures a dataset for the trainer (train or test) and makes sure it's valid."""
        
        # Configuring the Dataset
        if dataset_name:
            try:
                _dataset = DataSet(dataset_name, dataset_split_name)
                return _dataset.load()
            except Exception:
                err_msg = "Please provide a valid dataset split."
                if self.logger is not None:
                    self.logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif dataset:
            if isinstance(dataset, DataSet):
                try:
                    return dataset.load()
                except Exception:
                    err_msg = "Please provide a valid dataset split."
                    if self.logger is not None:
                        self.logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(dataset, HFDataset):
                return dataset
        
        else:
            err_msg = "Either dataset_name or dataset must be provided"
            if self.logger is not None:
                self.logger.exception(err_msg)
            raise ValueError(err_msg)
        
    
    def preprocessor(self, example, keys: list[str] = None):
        """
        Preprocess function for the dataset.
        """
        
        if keys:
            full_text = " ".join([str(value) for key, value in example.items() if (key in keys)])
        else:
            full_text = " ".join([str(value) for value in example.values()])
        
        try:
            encoded = self.model.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            encoded["labels"] = encoded["input_ids"].clone()
        except Exception as e:
            raise Exception(f"Failed to encode text '{full_text}': {str(e)}")
        
        return {k: v.squeeze() for k, v in encoded.items()}
    
    
    def save_model(self, output_dir: str = None):
        """
        Saves the model and tokenizer to the specified output directory.
        Args:
            output_dir (str): The directory where the model and tokenizer will be saved.
        """
        
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR + "/final"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {str(e)}")
        
        try:
            self.model.model.save_pretrained(output_dir)
            self.model.tokenizer.save_pretrained(output_dir)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to save model and tokenizer: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while saving the model: {str(e)}")
        
        if self.logger is not None:
            self.logger.info(f"Model and tokenizer saved to {output_dir}")
    
    
    def select_dataset_limit(self, dataset: HFDataset, limit: int):
        """Apply a limit to the dataset."""
        
        if limit <= 0 or limit > len(dataset):
            err_msg = f"Limit must be a positive integer less than or equal to the size of the given dataset. Current size: {len(dataset)}"
            if self.logger is not None:
                self.logger.exception(err_msg)
            raise ValueError(err_msg)
        
        if self.logger is not None:
            self.logger.info(f"Limiting dataset to {limit} samples.")
        return dataset.select(range(limit))
    
    
    def start_fine_tune(self, training_args: TrainingArguments = None, inplace: bool = False, limit_train: int = None, limit_test: int = None):
        """
        Fine-tunes the model using the provided dataset.

        Note:
            If inplace=False, a new model instance is created and returned (the original model remains untouched).
            If inplace=True, the self.text_gen.model is updated with fine-tuned weights.

        Args:
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            limit (int, optional): Limit the number of training samples. Defaults to None.
            inplace (bool, optional): Whether to fine-tune the model in-place or return a separate instance. Defaults to False.
        """

        base_model = self.model.model

        # Clone model if not inplace
        if not inplace:
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model.name_or_path)
                if self.logger is not None:
                    self.logger.info("Cloned model for isolated fine-tuning (inplace=False).")
            except Exception as e:
                err_msg = f"Failed to clone model for non-inplace fine-tuning: {str(e)}"
                if self.logger is not None:
                    self.logger.error(err_msg, exc_info=True)
                raise ValueError(err_msg)
        else:
            model = base_model

        try:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.model.tokenizer, mlm=False)
            if self.logger is not None:
                self.logger.info("Data collator created.")
        except Exception as e:
            raise ValueError(f"Failed to create data collator: {str(e)}")
        
        tokenized_train = self.tokenize_dataset(self.train_dataset, limit_train)
        if self.evaluation:
            tokenized_test = self.tokenize_dataset(self.test_dataset, limit_test)
        
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train, # always present
                eval_dataset=tokenized_test if self.evaluation else None, # specific to certain fine-tuning methods like SFT
                data_collator=data_collator,
            )
            if self.logger is not None:
                self.logger.info("Trainer initialized.")
        except Exception as e:
            err_msg = f"Failed to initialize Trainer: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg)
            raise ValueError(err_msg)

        try:
            if self.logger is not None:
                self.logger.info("Training started...")
            trainer.train()
            if self.logger is not None:
                self.logger.info("Training completed.")
        except Exception as e:
            err_msg = f"Training failed: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg)
            raise ValueError(err_msg)

        if inplace:
            self.model.model = model
            if self.logger is not None:
                self.logger.info("Model updated inplace.")
        
        return self.model


    def tokenize_dataset(self, dataset: HFDataset, limit: int = None):
        """Prepares the dataset for fine-tuning by tokenizing it."""
        
        # Limit dataset if needed
        dataset = self.select_dataset_limit(dataset, limit) if limit else dataset

        try:
            tokenized = dataset.map(
                self.preprocessor, remove_columns=dataset.column_names
            )
            if self.logger is not None:
                self.logger.info("Tokenized dataset.")
        except Exception as e:
            err_msg = f"Failed to tokenize dataset: {str(e)}"
            if self.logger is not None:
                self.logger.error(err_msg)
            raise ValueError(err_msg)

        return tokenized