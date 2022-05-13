import datasets
import os
import logging
from numpy import True_
import numpy as np
import re
import torch
import json
import copy

from pytorch_lightning import LightningDataModule
from torch.utils.data import (DataLoader, RandomSampler)
from transformers import (AutoTokenizer)

from torch.utils.data import random_split
from utils import (IterableDatasetPad,write_samples)
from data import qa as qa_processor
from datasets import load_dataset
logger = logging.getLogger(__name__)

class KorQuad2DataModule(LightningDataModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model_name_or_path = self.hparams.model_name_or_path
        self.output_dir = self.hparams.data_output_dir
        self.data_dir = self.hparams.data_dir
        self.max_seq_length = self.hparams.max_seq_length
        self.doc_stride = self.hparams.doc_stride
        self.max_query_length = self.hparams.max_query_length
        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size        
        version_2_with_negative =True
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        
        
    def setup(self,stage=None):
        self.dataset = {}
        train_dataset = self.get_data(is_train=True)
        train_len = int(len(train_dataset) * 0.8)
        val_len = len(train_dataset) - train_len
        self.dataset["train"], self.dataset["validation"] = random_split(train_dataset, [train_len,val_len])
        #self.dataset["test"], self.dataset["features"],  self.dataset["examples"] = self.load_and_cache_examples(self.tokenizer, evaluate=True, output_examples = True)

    def get_data(self, is_train=True, overwrite=False):
        if is_train:
            data_file = "train"
        else:
            data_file = "eval"

        data_path = self.data_dir
        if data_file is not None:
            data_path = os.path.join(data_path, data_file)
        else:
            data_path += "/"

        data_processor = qa_processor
        if data_processor is None:
            raise Exception(f"Invalid data task {self.hparams.task_name}!")
        processor = data_processor.process_map.get(self.hparams.data_file, None)

        comps = [
            data_path,
            self.hparams.data_file,
            self.model_name_or_path.replace("/", "_"),
            self.max_seq_length,
            "train" if is_train else "dev",
            "dataset.txt",
        ]
        dataset_file = "_".join([str(comp) for comp in comps])
        if not os.path.exists(dataset_file) or overwrite:
            with open(dataset_file, "w", encoding="utf-8") as writer_file:
                if data_file is None or not os.path.isdir(data_path):
                    data = processor(self.hparams, data_path, is_train)
                    cnt = write_samples(
                        self.hparams, self.tokenizer, is_train, data_processor, writer_file, data, workers=self.hparams.threads
                    )
                else:
                    cnt = 0
                    for filename in sorted([f for f in os.listdir(data_path) if f.endswith(".json")]):
                        data = processor(self.hparams, os.path.join(data_path, filename), is_train)
                        cnt += write_samples(
                             self.hparams, self.tokenizer, is_train, data_processor, writer_file, data, workers=self.hparams.threads
                        )
                logger.info(f"{cnt} features processed from {data_path}")

        dataset = load_dataset("text", data_files=dataset_file)["train"]

        dataset = dataset.map(lambda x: json.loads(x["text"]), batched=False)
        if not is_train:
            # for valid datasets, we pad datasets so that no sample will be skiped in multi-device settings
            dataset = IterableDatasetPad(
                dataset=dataset,
                batch_size=self.train_batch_size if is_train else self.eval_batch_size,
                num_devices=self.hparams.npgu,
                seed=self.hparams.seed,
            )
        return dataset
    
    def collate_fn(data_loader,features):
        input_ids = [sample["input_ids"] for sample in features]
        attention_mask = [sample["attention_mask"] for sample in features]
        token_type_ids = [sample["token_type_ids"] for sample in features]
        start_position = [sample["start_position"] for sample in features]
        end_position = [sample["end_position"] for sample in features]

        input_ids = torch.tensor(np.array(input_ids).astype(np.int64), dtype=torch.long)
        attention_mask = torch.tensor(np.array(attention_mask).astype(np.int8), dtype=torch.long)
        token_type_ids = torch.tensor(np.array(token_type_ids).astype(np.int8), dtype=torch.long)
        start_position = torch.tensor(np.array(start_position).astype(np.int64), dtype=torch.long)
        end_position = torch.tensor(np.array(end_position).astype(np.int64), dtype=torch.long)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_positions": start_position,
            "end_positions": end_position,
        }
        if "unique_id" in features[0]:
            inputs["unique_id"] = [sample["unique_id"] for sample in features]
        return inputs
    

    def train_dataloader(self):
        return DataLoader(self.dataset["train"],sampler= RandomSampler(self.dataset["train"]), 
                          drop_last=False, batch_size=self.train_batch_size, 
                          pin_memory=True, num_workers=4, collate_fn = self.collate_fn
                          )

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"],sampler= RandomSampler(self.dataset["validation"]), 
                          drop_last=False, batch_size=self.train_batch_size, 
                          pin_memory=True, num_workers=4, collate_fn = self.collate_fn
                          )
    '''
        return DataLoader(self.dataset["validation"],sampler= RandomSampler(self.dataset["validation"]), 
                          drop_last=False, batch_size=self.train_batch_size, 
                          pin_memory=True, num_workers=4, collate_fn= self.collate_fn
                          )

    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.train_batch_size, pin_memory=True, num_workers=4)
'''