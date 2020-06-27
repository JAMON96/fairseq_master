#!/bin/bash

MODEL=$1
BEAM=${2:-"1"}
WMT=${3:-"wmt14"}
LANGPAIR=${4:-"en-de"}
TASK=wmt16_en_de
FAIRSEQ_DATA="/media/drive/Datasets/fairseq-data"
DATA_DIR=$FAIRSEQ_DATA/$TASK
BPECODE="${DATA_DIR}/bpe.code"

SRCLANG=$(echo $LANGPAIR | cut -d '-' -f 1)
TGTLANG=$(echo $LANGPAIR | cut -d '-' -f 2)

# Compute BLEU score
sacrebleu -t wmt14 -l en-de --echo src
sacrebleu -t $WMT -l $LANGPAIR --echo src \
| fairseq-interactive $DATA_DIR --path $MODEL \
    -s $SRCLANG -t $TGTLANG \
    --cpu \
    --task prek_arrange_translation \
    --batch-size 8 --buffer-size 1024 \
    --beam $BEAM --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 \
    --bpe subword_nmt --bpe-codes $BPECODE --tokenizer moses --moses-no-dash-splits \
    --user-dir 'predk' \
| tee out.txt \
| grep ^H- | cut -f 3- \
| sacrebleu -t $WMT -l $LANGPAIR