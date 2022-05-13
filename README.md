# KoBigbird_lightning
Training KorQuAD2.0 by KoBigbird from huggingface🤗 and torch-lightning

## Navigation
1. [Objective]
2. [Quick start](#how-to-use)
3. [Pretraining using torch-lightning]


# Objective
Training KorQuAD2.0 dataset by using KoBigbird from huggingface🤗 and Pytorch-lightning

# Quick start
Please create a project first and create a file named 'output/'

```bash
mkdir output
```

And then run bash file train.sh for training KorQuAD1.0

```bash
bash train.sh
```

in bash file
```bash
#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="/media/storage/checkpoints/korbigbird/${DATE}/"
mkdir ${CKP_DIR}

python Trainer.py --model_name_or_path monologg/kobigbird-bert-base \
--max_seq_length 4096 \
--output_dir outputs/ \
--data_dir /media/storage/korquad/korquad_2/ \
--train_batch_size 1 \
--experiment_name $(date +%D-%T) \
--max_epochs 5 \
--learning_rate 3e-5 \
--ngpu 1 \
--warmup_steps 100 \
--task_name 'korbigbird' \
--checkpoints_dir ${CKPT_DIR} \
--data_file "korquad_2" \
--doc_stride 3072 \
--max_answer_length 4096 \
--gradient_accumulation_steps 4 \
```
make checkpoint file named date tha you run it

# Pretraining using torch-lightning
Torch-lightning models in 'models.py'
which is using KoBigbird from huggingface "monologg/kobigbird-bert-base"


KorQuAD 2.0 is preprocessed in 'dataloader.py'


I use 'Autotokenizer' for tokenization and preprocessing by squad preprocessing from huggingface


Not yet training because of resource problem


## Reference

https://github.com/monologg/KoBigBird