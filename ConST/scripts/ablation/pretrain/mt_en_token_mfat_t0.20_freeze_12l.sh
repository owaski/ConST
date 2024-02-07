#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=2
#SBATCH --partition=debug
#SBATCH --time=1-0:0:0
#SBATCH --account=siqiouyang
#SBATCH --mail-type=all
#SBATCH --mail-user=siqiouyang@ucsb.edu 
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

tag=mt_en_token_mfat_t0.20_freeze_12l

# pretrain_ckpt=wmt16_ende_xstnet_pretrain.pt
MODEL_DIR=/mnt/data3/siqiouyang/runs/ConST/$tag

mkdir -p ${MODEL_DIR}
# cp /mnt/data/siqiouyang/runs/ConST/pretrained/$pretrain_ckpt /mnt/data/siqiouyang/runs/ConST/$tag/checkpoint_last.pt

export num_gpus=2

python fairseq_cli/train.py /mnt/data/siqiouyang/datasets/must-c-v1.0 \
    --distributed-world-size $num_gpus \
    --task speech_to_text_triplet_align_with_extra_mt_nllb \
    --train-subset train_asr_mt_iwslt,train_asr_mt_cv --valid-subset dev_asr_mt_iwslt,dev_asr_mt_cv \
    --config-yaml config_st_mt_en.yaml \
    --langpairs mt-en --lang-prefix-tok "eng_Latn" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 800000 --max-text-tokens 2000 --max-tokens 800000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch xstnet_nllb_base --w2v2-model-path /mnt/data/siqiouyang/runs/mST/pretrained/xlsr2_300m.pt \
    --nllb-dir /mnt/data/siqiouyang/runs/ConST/pretrained/nllb --n-freeze-speech-encoder-layer 12 \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \--criterion multi_task_cross_entropy_with_contrastive_token_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 1.0 0.0 --contrastive-temperature 0.20 0.20 --contrastive-seqlen-type none --contrastive-level token \
    \
    --update-freq $(expr 20 / $num_gpus) --max-update 500000 \
    \
    --tensorboard-logdir tensorboard_logs/$tag --log-interval 100 \
    --save-interval-updates 1000 --save-interval 4 \
    --keep-last-epochs 1 --keep-interval-updates 1 --keep-best-checkpoints 1 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    --all-gather-list-size 32768 \
    --best-checkpoint-metric contrastive_loss \
    --reset-optimizer --reset-dataloader