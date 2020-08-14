#! /usr/bin/python3

import sys
import os
import argparse
import threading
import queue
import random
import tempfile
import inspect
import atexit
import numpy as np

from segment_generator import SegmentGenerator, PipeWriterThread
from noise_functions import noise_functions_dict

def main():
  arg_parser = argparse.ArgumentParser(description='serve noised monolingual data for semi-supervised pretraining of a Marian model')
  arg_parser.add_argument('src_pipe', type=str, help='named pipe file for the noised source segments')
  arg_parser.add_argument('tgt_pipe', type=str, help='named pipe file for the reconstruction target segments')
  arg_parser.add_argument('--corpora_filenames', nargs='*', default=[], help='monolingual corpora file names. If not provided reads from standard input. If multiple are provided cycles through for each segment')
  arg_parser.add_argument('--max_segments', type=str, default=-1, help='maximum number of pretraining segments (-1 = infinite, once = read the corpus once) (default -1)')
  arg_parser.add_argument('--buffer_size', type=int, default=10000, help='segment pair buffer size (default 10000 segments)')
  arg_parser.add_argument('--random_seed', type=int, default=None, help='random seed (default None)')
  arg_parser.add_argument('--shuffle_corpus', action='store_true', help='shuffle corpus between epochs. Ignored if reading from standard input')
  arg_parser.add_argument('--noise_function', type=str, choices=['copy', 'shuffle', 'bart'], default='copy', help='noise function used to generate segment pairs (default copy)')
  arg_parser.add_argument('--segment_size_in_sentences', type=int, default=1, help='number of sentences per segment, if --segment_size_in_tokens is undefined (default 1)')
  arg_parser.add_argument('--segment_size_in_tokens', type=int, default=-1, help='number of tokens per segment, overrides --segment_size_in_sentences')
  arg_parser.add_argument('--sentence_separator', type=str, default='<SEP>', help='token inserted to separate multiple sentences in a segment (default <SEP> )')
  arg_parser.add_argument('--noise_over_multiple_sentences', action='store_true', help='noise function does not adhere to sentence boundaries in a segment')
  arg_parser.add_argument('--noise_mask_token', type=str, default='<MASK>', help='mask token used by noise functions that mask text spans (default <MASK> )')
  arg_parser.add_argument('--noise_token_replacement_rate', type=float, default='.35', help='fraction of tokens replaced by the noise function (default .35)')
  arg_parser.add_argument('--noise_span_avg_length', type=float, default='3.5', help='average length of text spans replaced by the noise function (default 3.5)')
  arg_parser.add_argument('--special_prefix_tokens', nargs='*', default=[], help='special tokens to be preserved by the noise function when they appear at the beginning of a sentence (e.g. language id, domain id)')
  arg_parser.add_argument('--remove_special_prefix_tokens_from_target', action='store_true', help='remove the special prefix tokens specified by --special_prefix_tokens from the target segments')

  config = arg_parser.parse_args()
  config.script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
  config.full_sentence_separator = ' ' + config.sentence_separator + ' '
  config.num_corpora = len(config.corpora_filenames)
  assert ((not config.shuffle_corpus) or (config.num_corpora > 0)), '--shuffle_corpus requires at least one corpus file specified by --corpora_filenames'
  if config.max_segments == 'once':
    assert (config.num_corpora > 0), '--max_segments once requires at least one corpus file specified by --corpora_filenames'
    config.read_corpus_once = True
    config.max_segments_n = 0
  else:
    config.read_corpus_once = False
    config.max_segments_n = int(config.max_segments)
  assert ((not config.noise_over_multiple_sentences) or (len(special_prefix_tokens) == 0)), '--special_prefix_tokens can\'t be specified when --noise_over_multiple_sentences is enabled'
  print('Configuration:', config)
  config.noise_function_class = noise_functions_dict[config.noise_function]

  random.seed(config.random_seed)
  np.random.seed(random.randint(0, 2**32-1))
  state = SegmentGenerator(config)
  src_seg_thread = PipeWriterThread(state, 0, config.src_pipe)
  tgt_seg_thread = PipeWriterThread(state, 1, config.tgt_pipe)
  src_seg_thread.start()
  tgt_seg_thread.start()

  atexit.register(lambda : state.tempfile_cleanup())
  state.serve_segments()


if __name__ == '__main__':
  main()

