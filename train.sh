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