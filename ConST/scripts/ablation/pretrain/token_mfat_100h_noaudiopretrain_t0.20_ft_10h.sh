#!/usr/bin/env bash

tag=ablation_pretrain_token_mfat_100h_noaudiopretrain_t0.20_ft_10h

TGT_LANG=de
MODEL_DIR=/mnt/data/siqiouyang/runs/ConST/$tag

mkdir -p ${MODEL_DIR}
cp /mnt/data/siqiouyang/runs/ConST/ablation_pretrain_token_mfat_100h_noaudiopretrain_t0.20/checkpoint_best.pt /mnt/data/siqiouyang/runs/ConST/$tag/checkpoint_last.pt

export num_gpus=2

python fairseq_cli/train.py /mnt/data/siqiouyang/datasets/must-c-v1.0 \
    --distributed-world-size $num_gpus \
    --task speech_to_text_triplet_align_with_extra_mt \
    --train-subset train-10h_st --valid-subset dev_st_${TGT_LANG} \
    --config-yaml config_st_${TGT_LANG}.yaml \
    --langpairs en-${TGT_LANG} --lang-prefix-tok "<lang:${TGT_LANG}>" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch xstnet_base --w2v2-model-path /mnt/data/siqiouyang/runs/mST/pretrained/wav2vec_small.pt \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \
    --criterion multi_task_cross_entropy_with_contrastive_token_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 0.0 --contrastive-temperature 0.05 --contrastive-seqlen-type none --contrastive-level token \
    \
    --update-freq $(expr 16 / $num_gpus) --max-update 500000 \
    \
    --tensorboard-logdir tensorboard_logs/$tag --log-interval 100 \
    --save-interval-updates 1000 --save-interval 1 \
    --keep-last-epochs 1 --keep-interval-updates 1 --keep-best-checkpoints 1 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    --reset-optimizer --reset-dataloader  --all-gather-list-size 32768 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 4, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path /mnt/data/siqiouyang/datasets/must-c-v1.0/spm_unigram10000_st.model \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    # \
    # --external-parallel-mt-data extra_mt/bin/ --text-data-sample-ratio 0.25