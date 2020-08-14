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

noise_functions_dict = {}

class NoiseFunction(object):
  """
  Base class for segment noise generation
  """

  def __init__(self, config):
    self.config=config

  def apply_noise(self, segment):
    """
    Apply noise to a segment and return source and target versions.
    Target can be equal to the original segment, but not necessarily, depending on the noise function
    """
    raise NotImplementedError("apply_noise() not implemented")
    return "", ""

  def extract_special_prefix_tokens(self, sentence_tokens):
    """
    Utility function to extract special prefix tokens
    """
    if len(self.config.special_prefix_tokens) == 0:
      return [], sentence_tokens
    rv = []
    for i, token in enumerate(sentence_tokens):
      if token in self.config.special_prefix_tokens: 	# we assume that special_prefix_tokens is small, hence checking for membership is faster for an array than a set
        rv.append(token)
      else:
        break
    return rv, sentence_tokens[i:]

  def assemble_target(self, prefix_tokens, tokens):
    if self.config.remove_special_prefix_tokens_from_target:
      return ' '.join(tokens)
    else:
      return ' '.join(prefix_tokens + tokens)

class CopyNoiseFunction(NoiseFunction):
  """
  Simply return unmodified segments as both source and target
  Currey et al. 2017 "Copied Monolingual Data Improves Low-Resource Neural Machine Translation"
  """

  def __init__(self, config):
    super(CopyNoiseFunction, self).__init__(config)

  def apply_noise(self, segment):
    tgt_segment = self.assemble_target(*self.extract_special_prefix_tokens(segment.split()))
    return segment, segment

noise_functions_dict['copy'] = CopyNoiseFunction

class ShuffleNoiseFunction(NoiseFunction):
  """
  Source segment is randomly shuffled original segment, target is the same as the original
  """

  def __init__(self, config):
    super(ShuffleNoiseFunction, self).__init__(config)

  def apply_noise(self, segment):
    if self.config.noise_over_multiple_sentences:
      tokens = segment.split()
      src_segment = self.internal_apply_noise(tokens)
    else:
      src_segment_list = [self.internal_apply_noise(sentence) for sentence in segment.split(self.config.full_sentence_separator)]
      src_segment = self.config.full_sentence_separator.join(src_segment_list)
    return src_segment, segment

  def internal_apply_noise(self, x):
    tokens = x.split()
    prefix, tokens = self.extract_special_prefix_tokens(tokens)
    random.shuffle(tokens)
    return ' '.join(prefix + tokens)

noise_functions_dict['shuffle'] = ShuffleNoiseFunction

class BARTNoiseFunction(NoiseFunction):
  """
  Apply the BART noise function: gap text infilling and shuffling whole sentences in a segment
  Lewis et al. 2019 "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
  """

  def __init__(self, config):
    super(BARTNoiseFunction, self).__init__(config)

  def apply_noise(self, segment):
    if self.config.noise_over_multiple_sentences:
      tokens = segment.split()
      prefix, tokens = self.extract_special_prefix_tokens(tokens)
      src_segment = self.internal_apply_noise(prefix, tokens)
      tgt_segment = self.assemble_target(prefix, tokens)
    else:
      segment_list = segment.split(self.config.full_sentence_separator)
      src_segment_list = segment_list.copy()
      random.shuffle(src_segment_list) # whole sentence shuffling
      src_segment_list = [self.internal_apply_noise(*self.extract_special_prefix_tokens(sentence.split())) for sentence in src_segment_list]
      src_segment = self.config.full_sentence_separator.join(src_segment_list)
      tgt_segment_list = [self.assemble_target(*self.extract_special_prefix_tokens(sentence.split())) for sentence in segment_list]
      tgt_segment = self.config.full_sentence_separator.join(tgt_segment_list)
    return src_segment, tgt_segment

  def internal_apply_noise(self, prefix, tokens):
    #print(prefix, tokens, file=sys.stderr)
    num_tokens_to_replace = int(len(tokens) * self.config.noise_token_replacement_rate)
    rv_tokens = []
    last_i = 0
    just_added_empty_span=False
    while (num_tokens_to_replace > 0):
      span_length = np.random.poisson(self.config.noise_span_avg_length)
      if span_length > num_tokens_to_replace:
        span_length = num_tokens_to_replace
      i = np.random.randint(last_i, len(tokens) - num_tokens_to_replace)
      rv_tokens.extend(tokens[last_i:i])
      if (len(rv_tokens) == 0) or (rv_tokens[-1] != self.config.noise_mask_token):
        rv_tokens.append(self.config.noise_mask_token)
      last_i = i + span_length
      num_tokens_to_replace -= span_length
    rv_tokens.extend(tokens[last_i:])
    return ' '.join(prefix + rv_tokens)

noise_functions_dict['bart'] = BARTNoiseFunction

