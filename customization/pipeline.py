from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator, DataSet, random_hash
import torch
from datasets import Dataset as HFDataset
from typing import Any
import logging
import time
import os


OUTPUT_DIR = "./.cache/output"
LOGS_DIR = OUTPUT_DIR + "/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOGS_DIR + f"/log_{time.strftime('%Y%m%d_%H%M%S')}.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)


class DAFTTrainer:
    """
    Fine-tunes a given model using the Domain-Adaptive Fine-Tuning (DAFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """
    def __init__(
        self,
        text_gen: TextGenerator = None,
        model_name: str = None,
        dataset: DataSet | HFDataset | Any = None,
        dataset_name: str = None,
        dataset_split_name: str = "train",
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


    def fine_tune(self, save_to_disk: bool = False, output_dir: str = None, limit: int = None):
        """
        Fine-tunes the model using the provided dataset. (Domain-Adaptive Fine-Tuning **DAFT** approach)
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
                    output_dir = OUTPUT_DIR + f"/daft/daft_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                logger.info(f"save_to_disk was set to 'True'. Output directory set to {output_dir}.")
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=4,
                    save_strategy="epoch",
                    num_train_epochs=3,
                    learning_rate=5e-5,
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    logging_dir=f"{output_dir}/logs",
                    report_to="none",
                )
            else:
                # in-memory training without disk I/O
                training_args = TrainingArguments(
                    output_dir=OUTPUT_DIR,  # required but won't be used
                    per_device_train_batch_size=4,
                    save_strategy="no",
                    num_train_epochs=3,
                    learning_rate=5e-5,
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    report_to="none",
                    dataloader_pin_memory=False,
                    remove_unused_columns=True,
                )
            logger.info(f"Training arguments set up.")
        except Exception as e:
            err_msg = f"Failed to set up training arguments: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                data_collator=data_collator,
            )
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
            output_dir = OUTPUT_DIR + "/final_daft"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.text_gen.model.save_pretrained(output_dir)
        self.text_gen.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}.")


class SFTTrainer:
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
        
        # Configuring the Training Dataset
        if train_dataset_name:
            try:
                _dataset = DataSet(train_dataset_name, train_dataset_split_name)
                self.train_dataset = _dataset.load()
            except Exception:
                err_msg = "Please provide a valid training dataset split."
                logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif train_dataset:
            if isinstance(train_dataset, DataSet):
                try:
                    self.train_dataset = train_dataset.load()
                except Exception:
                    err_msg = "Please provide a valid training dataset split."
                    logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(train_dataset, HFDataset):
                self.train_dataset = train_dataset # assuming it's good to go :3
        
        else:
            err_msg = "Either train_dataset_name or train_dataset must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)
        
        # Configuring the Test Dataset
        if test_dataset_name:
            try:
                _dataset = DataSet(test_dataset_name, test_dataset_split_name)
                self.test_dataset = _dataset.load()
            except Exception:
                err_msg = "Please provide a valid test dataset split."
                logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif test_dataset:
            if isinstance(test_dataset, DataSet):
                try:
                    self.test_dataset = test_dataset.load()
                except Exception:
                    err_msg = "Please provide a valid test dataset split."
                    logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(test_dataset, HFDataset):
                self.test_dataset = test_dataset # assuming it's good to go :3
        
        else:
            err_msg = "Either test_dataset_name or test_dataset must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)


    def fine_tune(self, save_to_disk: bool = False, output_dir: str = None, train_limit: int = None, test_limit: int = None):
        """
        Fine-tunes the model using the provided dataset. (Supervised Fine-Tuning **SFT** approach)
        #### Note:
        This method will change the model's parameters **inplace**.
        Args:
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
            output_dir (str, optional): The directory where the model and tokenizer will be saved. Defaults to None.
            train_limit (int, optional): Limit the number of training samples. Defaults to None.
            test_limit (int, optional): Limit the number of test samples. Defaults to None.
        """
        model = self.text_gen.model
        tokenizer = self.text_gen.tokenizer
        _train_dataset = self.train_dataset
        _test_dataset = self.test_dataset
        
        if train_limit:
            if train_limit <= 0 or train_limit > len(self.train_dataset):
                err_msg = f"Limit must be a positive integer less than or equal to the size of the datasets. Current sizes: train={len(self.train_dataset)}, test={len(self.test_dataset)}"
                logger.exception(err_msg)
                raise ValueError(err_msg)
            logger.info(f"Limiting training dataset to {train_limit} samples.")
            _train_dataset = self.train_dataset.select(range(train_limit))
        
        if test_limit:
            if test_limit <= 0 or test_limit > len(self.test_dataset):
                err_msg = f"Limit must be a positive integer less than or equal to the size of the datasets. Current sizes: train={len(self.train_dataset)}, test={len(self.test_dataset)}"
                logger.exception(err_msg)
                raise ValueError(err_msg)
            logger.info(f"Limiting test dataset to {test_limit} samples.")
            _test_dataset = self.test_dataset.select(range(test_limit))

        # Apply the preprocessor to the datasets
        try:
            tokenized_train = _train_dataset.map(
                self.preprocessor, remove_columns=_train_dataset.column_names
            )
            logger.info(f"Tokenized training dataset.")
        except Exception as e:
            err_msg = f"Failed to tokenize training dataset: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            tokenized_test = _test_dataset.map(
                self.preprocessor, remove_columns=_test_dataset.column_names
            )
            logger.info(f"Tokenized test dataset.")
        except Exception as e:
            err_msg = f"Failed to tokenize test dataset: {str(e)}"
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
                    output_dir = OUTPUT_DIR + f"/sft/sft_session_{_hash}"
                    os.makedirs(output_dir, exist_ok=True)
                logger.info(f"save_to_disk was set to 'True'. Output directory set to {output_dir}.")
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    num_train_epochs=3,
                    learning_rate=5e-5,
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    logging_dir=f"{output_dir}/logs",
                    report_to="none",
                )
            else:
                # in-memory training without disk I/O
                training_args = TrainingArguments(
                    output_dir=OUTPUT_DIR,  # required but won't be used
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    eval_strategy="epoch",
                    save_strategy="no",
                    num_train_epochs=3,
                    learning_rate=5e-5,
                    warmup_steps=100,
                    weight_decay=0.01,
                    fp16=torch.cuda.is_available(),
                    report_to="none",
                    dataloader_pin_memory=False,
                    remove_unused_columns=True,
                )
            logger.info(f"Training arguments set up.")
        except Exception as e:
            err_msg = f"Failed to set up training arguments: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                data_collator=data_collator,
            )
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
            return_tensors="pt"
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
            output_dir = OUTPUT_DIR + "/final_sft"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.text_gen.model.save_pretrained(output_dir)
        self.text_gen.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}.")