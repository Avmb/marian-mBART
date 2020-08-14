#/bin/bash -v

MARIAN=/path-to-marian
SENTENCEPIECE=/path-to-sentencepiece

# if we are in WSL, we need to add '.exe' to the tool names
if [ -e "/bin/wslpath" ]
then
    EXT=.exe
fi

MARIAN_TRAIN=$MARIAN/marian$EXT
MARIAN_DECODER=$MARIAN/marian-decoder$EXT
MARIAN_VOCAB=$MARIAN/marian-vocab$EXT
MARIAN_SCORER=$MARIAN/marian-scorer$EXT

# set chosen gpus
GPUS=0
if [ $# -ne 0 ]
then
    GPUS=$@
fi
echo Using GPUs: $GPUS


WORKSPACE=9500
N=4
EPOCHS=32
B=12

DATA_DIR=/path-to-data
VALID_CORPUS=$DATA_DIR/validation_corpus_with_static_mbart_noise_basename
VALID_CORPUS_SRC=$VALID_CORPUS.src
VALID_CORPUS_TGT=$VALID_CORPUS.tgt
VALID_CORPUS_TGT_RAW=$VALID_CORPUS_TGT.debpe

TRAIN_CORPUS_SRC=../src_pipe_sp20000_no_dedup
TRAIN_CORPUS_TGT=../tgt_pipe_sp20000_no_dedup

VOCAB=$DATA_DIR/spm.20000.vocab.yml

if [ ! -e $MARIAN_TRAIN ]
then
    echo "marian is not installed in $MARIAN, you need to compile the toolkit first"
    exit 1
fi

if [ ! -e ../tools/moses-scripts ] || [ ! -e ../tools/subword-nmt ] || [ ! -e ../tools/sacreBLEU ]
then
    echo "missing tools in ../tools, you need to download them first"
    exit 1
fi


  # train model
    $MARIAN_TRAIN \
        --model model/model.npz --type transformer \
        --task transformer-base \
        --train-sets $TRAIN_CORPUS_SRC $TRAIN_CORPUS_TGT \
        --max-length 512 \
        --vocabs $VOCAB $VOCAB \
        --mini-batch-fit -w $WORKSPACE \
        --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
        --valid-metrics ce-mean-words perplexity translation \
        --valid-sets $VALID_CORPUS_SRC $VALID_CORPUS_TGT \
        --beam-size 6 --normalize=1 \
        --valid-mini-batch 8 \
        --overwrite --keep-best \
        --early-stopping 10 --after-epochs $EPOCHS --cost-type=ce-mean-words \
        --log model/train.log --valid-log model/valid.log \
        --tied-embeddings-all \
        --optimizer-delay 2 \
        --sync-sgd \
        --devices $GPUS --seed 42  \
        --exponential-smoothing \
        --shuffle none \
        --maxi-batch-sort none \
        --no-restore-corpus \
        --valid-script-path "bash ./validate.sh $VALID_CORPUS_TGT_RAW" \
        --valid-translation-output ./valid.bpe.tgt.output --quiet-translation \

# translate parallel valid sets
PARALLEL_VALID_EN_SRC=$DATA_DIR/newsdev2020.en-ta.sp20000.domain-tag.out-lang-tag.en
PARALLEL_VALID_EN_TGT=$DATA_DIR/newsdev2020.en-ta.ta
PARALLEL_VALID_TA_SRC=$DATA_DIR/newsdev2020.en-ta.sp20000.domain-tag.out-lang-tag.ta
PARALLEL_VALID_TA_TGT=$DATA_DIR/newsdev2020.en-ta.en

    cat $PARALLEL_VALID_EN_SRC \
        | $MARIAN_DECODER -c model/model.npz.best-ce-mean-words.npz.decoder.yml  \
          -m model/model.npz.best-ce-mean-words.npz -d $GPUS \
          --mini-batch 16 --maxi-batch 100 --maxi-batch-sort src -w 5000 --beam-size $B  --quiet-translation \
        >  ./dev.en.output.zeroshot

    cat $PARALLEL_VALID_TA_SRC \
        | $MARIAN_DECODER -c model/model.npz.best-ce-mean-words.npz.decoder.yml  \
          -m model/model.npz.best-ce-mean-words.npz -d $GPUS \
          --mini-batch 16 --maxi-batch 100 --maxi-batch-sort src -w 5000 --beam-size $B  --quiet-translation \
        >  ./dev.ta.output.zeroshot

SENTENCEPIECE_MODEL=$DATA_DIR/spm.20000.model

    $SENTENCEPIECE/spm_decode --model=$SENTENCEPIECE_MODEL < ./dev.en.output.zeroshot > ./dev.en.output.zeroshot.decoded
    $SENTENCEPIECE/spm_decode --model=$SENTENCEPIECE_MODEL < ./dev.ta.output.zeroshot > ./dev.ta.output.zeroshot.decoded

# calculate bleu scores on test sets
LC_ALL=C.UTF-8 ../tools/sacreBLEU/sacrebleu.py $PARALLEL_VALID_EN_TGT -l en-ta < ./dev.en.output.zeroshot.decoded
LC_ALL=C.UTF-8 ../tools/sacreBLEU/sacrebleu.py $PARALLEL_VALID_TA_TGT -l ta-en < ./dev.ta.output.zeroshot.decoded


