import os
import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import (
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForQuestionAnswering,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
import torchmetrics

MODEL_CLASSES = {
    "kobert": (BertConfig, BertForQuestionAnswering),
    "korbigbird" : (None, AutoModelForQuestionAnswering)
}

class Korbigbird(LightningModule):
    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__()
        # Save Hyper parameters
        self.save_hyperparameters(hparams)
        # version difference
        #self.hparams.update(hparams)
        self.task_name = self.hparams.task_name
        self.learning_rate= self.hparams.learning_rate
        self.adam_epsilon= self.hparams.adam_epsilon
        self.warmup_steps= self.hparams.warmup_steps
        self.weight_decay= self.hparams.weight_decay
        self.train_batch_size= self.hparams.train_batch_size
        self.eval_batch_size= self.hparams.eval_batch_size
        self.eval_splits= self.hparams.eval_splits
        
        self.accuracy = torchmetrics.Accuracy()
        config_type,model_type = MODEL_CLASSES[self.task_name]

        #Configuration
        #self.config = config_type.from_pretrained(self.hparams.model_name_or_path)
        
        self.model = model_type.from_pretrained(self.hparams.model_name_or_path,
                                                from_tf=bool(".ckpt" in hparams.model_name_or_path),
                                                cache_dir=self.hparams.checkpoints_dir
                                                )

    def forward(self, **inputs):
        return self.model(**inputs)


    def training_step(self,  batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }
        outputs = self(**inputs)# (loss, start_logits, end_logits)
        start_preds = outputs[1].argmax(dim=-1)
        start_positions = inputs["start_positions"]
        end_preds = outputs[2].argmax(dim=-1)
        end_positions = inputs["end_positions"]
        
        acc = (self.accuracy(start_preds, start_positions) + self.accuracy(end_preds, end_positions)) / 2
        loss = outputs[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }
        outputs = self(**inputs)# (loss, start_logits, end_logits)
        start_preds = outputs[1].argmax(dim=-1)
        start_positions = inputs["start_positions"]
        end_preds = outputs[2].argmax(dim=-1)
        end_positions = inputs["end_positions"]
        acc = (self.accuracy(start_preds, start_positions) + self.accuracy(end_preds, end_positions)) / 2
        loss =outputs[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return outputs[0]
    
    def test_step(self,batch,batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        result_list = []
        example_indices = batch[3]
        outputs = self(**inputs)# (start_logits, end_logits)
        for i, example_index in enumerate(example_indices):
            eval_feature = self.val_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = (outputs[0][i].detach().cpu().tolist(), outputs[1][i].detach().cpu().tolist())
            result = SquadResult(unique_id, start_logits, end_logits)
            result_list.append(result)
            
        return {"results" : result_list}
    
    
    def test_step_end(self, batch_parts):
        self.all_results = self.all_results + batch_parts["results"]
        return self.all_results
        
        
    def test_epoch_end(self, outputs):
        self.pred_file()
        predictions = compute_predictions_logits(
            self.val_examples,
            self.val_features,
            self.all_results,
            self.eval_batch_size,
            self.hparams.max_answer_length,
            self.hparams.do_lower_case,
            self.output_prediction_file,
            self.output_nbest_file,
            self.output_null_log_odds_file,
            self.hparams.verbose_logging,
            self.hparams.version_2_with_negative,
            self.hparams.null_score_diff_threshold,
            self.tokenizer,
        )
        results = squad_evaluate(self.val_examples, predictions)
        
        return results
    
    def pred_file(self,prefix=''):
        self.output_prediction_file = os.path.join(self.hparams.output_dir, "predictions_{}.json".format(prefix))
        self.output_nbest_file = os.path.join(self.hparams.output_dir, "nbest_predictions_{}.json".format(prefix))
        if self.hparams.version_2_with_negative:
            self.output_null_log_odds_file = os.path.join(self.hparams.output_dir, "null_odds_{}.json".format(prefix))
        else:
            self.output_null_log_odds_file = None
            
    def setup(self,stage):
        if stage in (None, "test"):
            self.val_features = self.trainer.datamodule.dataset["features"]
            self.val_examples = self.trainer.datamodule.dataset["examples"] 
            self.all_results = []
        self.tokenizer = self.trainer.datamodule.tokenizer
        train_loader = self.trainer.datamodule.train_dataloader()
        
        # Setting
        tb_size = self.hparams.train_batch_size  * self.trainer.accumulate_grad_batches * max(1, self.trainer.gpus)
        self.total_steps = (len(train_loader.dataset) // tb_size) * self.trainer.max_epochs
        self.warmup_steps = int(len(train_loader.dataset) / self.trainer.gpus * self.trainer.max_epochs * 0.2)


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]