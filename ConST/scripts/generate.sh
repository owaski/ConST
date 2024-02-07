#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=1-0:0:0
#SBATCH --account=siqiouyang
#SBATCH --mail-type=all
#SBATCH --mail-user=siqiouyang@ucsb.edu 
#SBATCH --output=generate_stdout.txt
#SBATCH --error=generate_stderr.txt

split=tst-COMMON_st_de
name=ablation_pretrain_token_mfat_10h_noaudiopretrain_ft_1h

fairseq-generate /mnt/data/siqiouyang/datasets/must-c-v1.0/ --gen-subset $split --task speech_to_text --prefix-size 1 \
--max-tokens 4000000 --max-source-positions 4000000 --beam 10 --lenpen 0.6 --scoring sacrebleu \
--config-yaml config_st_de.yaml  --path /mnt/data/siqiouyang/runs/ConST/$name/checkpoint_best.pt \
--results-path /home/siqiouyang/work/projects/ConST/ConST/analysis/generation/$name # \
# --mt-mode 