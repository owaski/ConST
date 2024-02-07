import os
import random
from string import punctuation

from tqdm import tqdm

import transformers
import sentencepiece
import torchaudio
import textgrids

import numpy as np
import torch as th
import pandas as pd

from g2p_en import G2p

from fairseq.data import PhonemeDictionary
from ConST.prepare_data.data_utils import load_df_from_tsv, save_df_to_tsv

root = '/mnt/data/siqiouyang/datasets/must-c-v1.0/'
lang = 'es'
# spm = sentencepiece.SentencePieceProcessor(os.path.join(root, 'spm_unigram10000_st_{}.model'.format(lang)))
# spm = sentencepiece.SentencePieceProcessor(os.path.join(root, 'flores200sacrebleuspm.model'))
tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/mnt/taurus/data/xixu/llm/llama-2-7b/hf',
    padding_side="right",
    use_fast=False,
)
split = 'train'
df = load_df_from_tsv(os.path.join(root, '{}_st_{}.tsv'.format(split, lang)))
# df = load_df_from_tsv(os.path.join(root, '{}_st_{}.tsv'.format(split)))
# df = load_df_from_tsv(os.path.join(root, 'train_asr_mt_iwslt.tsv'))
# df2 = load_df_from_tsv(os.path.join(root, 'dev_asr_mt_iwslt.tsv'))
# df3 = load_df_from_tsv(os.path.join(root, 'train_asr_mt_cv.tsv'))
# df4 = load_df_from_tsv(os.path.join(root, 'dev_asr_mt_cv.tsv'))
# df2 = load_df_from_tsv(os.path.join(root, 'dev_st_id_en.tsv'))
# df3 = load_df_from_tsv(os.path.join(root, 'test_st_id_en.tsv'))
# df = pd.concat([df, df2, df3, df4], ignore_index=True)
df
df['n_frames'].sum() / 16000 / 3600
# n_frames = 0
# for fn in os.listdir('/mnt/data/siqiouyang/datasets/must-c-v1.0/en-de/data/dev/wav'):
#     info = torchaudio.info('/mnt/data/siqiouyang/datasets/must-c-v1.0/en-de/data/dev/wav/' + fn)
#     n_frames += info.num_frames
# n_frames / 16000 / 3600

# import yaml
# with open('/mnt/data/siqiouyang/datasets/must-c-v1.0/en-de/data/train/txt/train.yaml') as r:
#     y = yaml.load(r, Loader=yaml.Loader)

# duration = 0
# for x in y:
#     duration += x['duration']
# duration / 3600
# indices = list(range(train_df.shape[0]))
# random.shuffle(indices)
# save_df_to_tsv(train_df.iloc[indices[:10000]], os.path.join(root, 'train-tiny_asr.tsv'))
save_dir = os.path.join(root, 'en-{}'.format(lang), 'data', split, 'align_sllama')
# save_dir = os.path.join(root, 'mt-en', 'data', 'asr', 'align_mfat')
# save_dir = os.path.join(root, 'data', split, 'align_sllama')
os.makedirs(save_dir, exist_ok=True)
save_dir
last_audio_path = None
for idx in tqdm(range(len(df))):
    audio_path, offset, num_frames = os.path.join(root, df['audio'][idx]).split(':')
    # audio_path = os.path.join(root, df['audio'][idx])
    # offset, num_frames = 0, df['n_frames'][idx]
    offset, num_frames = int(offset), int(num_frames)
    if last_audio_path is None or audio_path != last_audio_path:
        waveform, frame_rate = torchaudio.load(os.path.join(root, audio_path))
        last_audio_path = audio_path
    torchaudio.save(os.path.join(save_dir, '{}.wav'.format(df['id'][idx])), waveform[:, offset : offset + num_frames], sample_rate=frame_rate)
sentences = df['src_text'].tolist()
# punctuation = '!"#$%&,.?' # for maltese
def covered(s, punctuation):
    for c in s:
        if c not in punctuation:
            return False
    return True

space = '▁'
tokenized_sentences = []
segmentss = []
punctuation = punctuation
for sent in tqdm(df['src_text'].tolist()):
    # tokens = spm.EncodeAsPieces(sent)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sent, add_special_tokens=False))
    segments = []
    last = -1
    for idx, token in enumerate(tokens):
        if token.startswith(space) or covered(token, punctuation):
            if last != -1 and last <= idx - 1:
                segments.append((last, idx - 1))
            last = idx + (token == space or covered(token, punctuation) or \
                (token.startswith(space) and len(token) > 1 and covered(token[1:], punctuation)))    
    
    if last < len(tokens):
        segments.append((last, len(tokens) - 1))

    tokenized_sentence = []
    for seg in segments:
        token = ''.join(tokens[seg[0] : seg[1] + 1]).replace(space, '')
        if token.replace(',', '').isnumeric():
            token = token.replace(',', '')
        tokenized_sentence.append(token)

    tokenized_sentences.append(tokenized_sentence)
    segmentss.append(segments)
for i, id in enumerate(tqdm(df['id'])):
    # if id.startswith('common_voice'):
    #     write_dir = os.path.join(save_dir, df.iloc[i]['speaker'])
    # else:
    #     write_dir = os.path.join(save_dir, id[:id.rfind('_')])
    write_dir = save_dir 
    with open(os.path.join(write_dir, '{}.txt'.format(id)), 'w') as w:
        w.write(' '.join(tokenized_sentences[i]))