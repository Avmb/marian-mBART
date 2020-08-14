#! /bin/sh

DATA_DIR=/path-to-data
SCRIPT_DIR=../
TRAIN_CORPUS=$DATA_DIR/shuffled_training_corpus_basename
TRAIN_CORPUS_0=$TRAIN_CORPUS.en
TRAIN_CORPUS_1=$TRAIN_CORPUS.ta


$SCRIPT_DIR/unsup_pretrain.py ./src_pipe_sp20000 ./tgt_pipe_sp20000  --noise_function bart --segment_size_in_sentences 1 --corpora_filenames $TRAIN_CORPUS_0 $TRAIN_CORPUS_1 --random_seed 42 --remove_special_prefix_tokens_from_target --special_prefix_tokens "<domain:news>" "<domain:other>" "<lang:en>" "<lang:ta>" --sentence_separator "<sep>" --noise_mask_token "<mask>" 
