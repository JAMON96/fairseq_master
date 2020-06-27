TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
EXPERIMENT=${1:-$TIMESTAMP}
TASK=wmt16_en_de
FAIRSEQ_DATA="/media/drive/Datasets/fairseq-data"
RAW_DATA_DIR="/media/drive/Datasets/${TASK}" # Download from https://drive.google.com/uc?id=0B_bZck-ksdkpM25jRUN2X2UxMm8&export=download
DATA_DIR=$FAIRSEQ_DATA/$TASK

cp "${RAW_DATA_DIR}/bpe.32000" "${DATA_DIR}/bpe.code"

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $RAW_DATA_DIR/train.tok.clean.bpe.32000 \
    --validpref $RAW_DATA_DIR/newstest2013.tok.bpe.32000 \
    --testpref $RAW_DATA_DIR/newstest2014.tok.bpe.32000 \
    --destdir $DATA_DIR \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

TOTAL_UPDATES=50000     # Total number of training steps
WARMUP_UPDATES=4000     # Warmup the learning rate over this many updates
PEAK_LR=1e-3            # Peak learning rate, adjust as needed
UPDATE_FREQ=16          # Increase the batch size Nx
SAVE_INTERVAL=200       # Save every N iterations
KEEP_CHECKPOINTS=1      # Keep past N checkpoints
SAVE_DIR="./results/checkpoints/${EXPERIMENT}"
TENSORBOARD_DIR="./results/tensorboard/${EXPERIMENT}"
LOG="${SAVE_DIR}/log.txt"

mkdir -p $SAVE_DIR
cp "${RAW_DATA_DIR}/bpe.32000" "${SAVE_DIR}/bpe"

fairseq-train $DATA_DIR \
    --user-dir 'permuted' \
    --task permuted_translation --max-shuffle-distance 8 \
    --arch transformer_permuted --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --max-update $TOTAL_UPDATES \
    --lr $PEAK_LR --lr-scheduler inverse_sqrt --warmup-updates $WARMUP_UPDATES --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ --max-tokens-valid 256 \
    --eval-bleu --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates $SAVE_INTERVAL --keep-interval-updates $KEEP_CHECKPOINTS \
    --no-epoch-checkpoints --log-format simple --log-interval 10 \
    --save-dir $SAVE_DIR --tensorboard-logdir $TENSORBOARD_DIR | tee -a $LOG



#fairseq-train /media/drive/Datasets/fairseq-data/wmt16_en_de
#    --user-dir 'permuted'
#    --task permuted_translation --max-shuffle-distance 8 --mark-end --augment-reverse
#    --arch transformer_permuted --share-all-embeddings
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --max-update 50000
#    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07
#    --dropout 0.1 --weight-decay 0.0
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1
#    --max-tokens 4096 --update-freq 16
#    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'
#    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
#    --save-interval-updates 200 --keep-interval-updates 1
#    --no-epoch-checkpoints --log-format simple --log-interval 10
#    --save-dir ./results/checkpoints/test --tensorboard-logdir ./results/tensorboard/test