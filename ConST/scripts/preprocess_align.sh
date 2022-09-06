conda activate st
python prepare_mfa.py --data-root /mnt/data/siqiouyang/datasets/must-c-v1.0 --lang de

conda activate mfa

mfa models download acoustic english_mfa
mfa models download dictionary english_mfa

mfa align /mnt/data/siqiouyang/datasets/must-c-v1.0/en-de/data/train/align english_mfa english_mfa /mnt/data/siqiouyang/datasets/must-c-v1.0/en-de/data/train/align/textgrids/

conda activate st
python finish_mfa.py --data-root /mnt/data/siqiouyang/datasets/must-c-v1.0 --lang de