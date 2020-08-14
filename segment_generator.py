import sys
import os
import argparse
import threading
import queue
import random
import tempfile
import inspect
import atexit

class SegmentGenerator(object):
  def __init__(self, config):
    self.config=config
    seg_src_queue = queue.Queue()
    seg_tgt_queue = queue.Queue()
    self.seg_queues = [seg_src_queue, seg_tgt_queue]
    self.pipe_writers_cv = threading.Condition()
    self.noise_function=self.config.noise_function_class(config)
    self.keep_running=True
    self.num_sentences=0
    self.num_segments=0
    self.corpus_in_fs_list=[None] * config.num_corpora
    self.tempfile_path_list=[None] * config.num_corpora
    self.buffered_sentence_list=[None] * config.num_corpora
    self.active_corpus_id = 0

  def serve_segments(self):
    segment_buffer = []
    while self.keep_running:
      with self.pipe_writers_cv:
        cur_min_queue_size = min([q.qsize() for q in self.seg_queues])
        if cur_min_queue_size >= self.config.buffer_size:
          self.pipe_writers_cv.wait()
        segment_pair = self.generate_segment_pair()
        if segment_pair == None:
          self.keep_running = False
        else:
          for i, segment in enumerate(segment_pair):
            self.seg_queues[i].put(segment)
        self.pipe_writers_cv.notify_all()

  def generate_segment_pair(self):
    if (self.config.max_segments_n > 0) and (self.num_segments >= self.config.max_segments_n):
      return None
    segment = self.read_segment()
    if segment == "":
      return None
    segment = segment.strip()
    self.num_segments += 1

    src_segment, tgt_segment = self.noise_function.apply_noise(segment)
    return (src_segment, tgt_segment)

  def read_sentence(self):
    try:
      if self.config.num_corpora == 0:
        # read from stdin
        line = sys.stdin.readline()
      elif not self.config.shuffle_corpus:
        # no shuffling, we just read from the file and reset the file stream at the end
        line = ""
        while line == "":
          if self.corpus_in_fs_list[self.active_corpus_id] == None:
            self.corpus_in_fs_list[self.active_corpus_id] = open(self.config.corpora_filenames[self.active_corpus_id])
            print('Opened corpus file %s' % self.config.corpora_filenames[self.active_corpus_id], file=sys.stderr)
          line = self.corpus_in_fs_list[self.active_corpus_id].readline()
          if line == "":
            if self.config.read_corpus_once:
              break
            self.corpus_in_fs_list[self.active_corpus_id].seek(0)
      else:
        line = ""
        while line == "":
          if self.corpus_in_fs_list[self.active_corpus_id] == None:
            self.shuffle_corpus()
          line = self.corpus_in_fs_list[self.active_corpus_id].readline()
          if line == "":
            self.corpus_in_fs_list[self.active_corpus_id].close()
            self.corpus_in_fs_list[self.active_corpus_id] = None
            os.remove(self.tempfile_path_list[self.active_corpus_id])
            if self.config.read_corpus_once:
              break
      self.num_sentences += 1
      return line
    except OSError as oserror:
      # allow the pipewriter threads to terminate gracefully after exhausting their buffers
      return ''

  def read_segment(self):
    segment_acc = []

    if (self.config.segment_size_in_tokens == -1):
      # count by sentences
      for sentence_count in range(self.config.segment_size_in_sentences):
        cur_sentence = self.read_sentence()
        if cur_sentence == "":
          break
        cur_sentence = cur_sentence.strip()
        segment_acc.append(cur_sentence)
    else:
      # count by tokens
      token_count = 0
      while True:
        cur_sentence = self.read_sentence() if (self.buffered_sentence_list[self.active_corpus_id] == None) else self.buffered_sentence_list[self.active_corpus_id]
        self.buffered_sentence_list[self.active_corpus_id] = None
        if cur_sentence == "":
          break
        cur_sentence = cur_sentence.strip()
        cur_tokens = cur_sentence.split()
        new_token_count = token_count + len(cur_tokens) + 1 	# also count the separator
        if new_token_count < self.config.segment_size_in_tokens:
          segment_acc.append(cur_sentence)
          token_count = new_token_count
        else:
          self.buffered_sentence_list[self.active_corpus_id] = cur_sentence
          break
    if self.config.num_corpora > 0:
      # cycle corpora
      self.active_corpus_id = (self.active_corpus_id + 1) % self.config.num_corpora
    segment_str = self.config.full_sentence_separator.join(segment_acc)
    return segment_str
    

  def shuffle_corpus(self):
    # shuffle on temporary file
    temp_out_fd, self.tempfile_path_list[self.active_corpus_id] = tempfile.mkstemp(prefix='unsup_pretrain_temp_', text=True)
    os.close(temp_out_fd)
    print('Shuffling corpus file %s to tempfile %s' % (self.config.corpora_filenames[self.active_corpus_id], self.tempfile_path_list[self.active_corpus_id]), file=sys.stderr)
    shuffle_script_path = os.path.join(self.config.script_dir, 'shuffle_corpus.sh')
    shuffle_seed = str(random.randint(0, 2**64-1))
    shuffle_rv = os.system('%s %s < %s > %s' % (shuffle_script_path, shuffle_seed, self.config.corpora_filenames[self.active_corpus_id], self.tempfile_path_list[self.active_corpus_id]))
    if shuffle_rv != 0:
      # shuffling failed
      raise OSError()
    self.corpus_in_fs_list[self.active_corpus_id] = open(self.tempfile_path_list[self.active_corpus_id])
    print('Opened tempfile file %s' % self.tempfile_path_list[self.active_corpus_id], file=sys.stderr)

  def tempfile_cleanup(self):
    for i in range(self.config.num_corpora):
      if (self.corpus_in_fs_list[i] != None) and (not self.corpus_in_fs_list[i].closed):
        self.corpus_in_fs_list[i].close()
      if (self.tempfile_path_list[i] != None) and (os.path.isfile(self.tempfile_path_list[i])):
        os.remove(self.tempfile_path_list[i])

class PipeWriterThread(threading.Thread):
  def __init__(self, state, id_in_tuple, pipe_filename):
    super(PipeWriterThread, self).__init__(daemon=False)
    self.state=state
    self.id_in_tuple=id_in_tuple
    self.pipe_filename=pipe_filename

  def run(self):
    while True:
      try:
        with open(self.pipe_filename, 'w') as pipe_out_fs:
          print('Opened pipe %s' % self.pipe_filename, file=sys.stderr)
          while True:
            with self.state.pipe_writers_cv:
              self.state.pipe_writers_cv.notify_all()
              try:
                segment = self.state.seg_queues[self.id_in_tuple].get_nowait()
              except queue.Empty as empty:
                segment = None
                pipe_out_fs.flush()
                if not self.state.keep_running:
                  return
                self.state.pipe_writers_cv.wait()
            if segment != None:
              print(segment, file=pipe_out_fs)
      except OSError as oserror:
        pass

