

```python
!curl http://mattmahoney.net/dc/text8.zip > text8.zip
!unzip text8.zip
!curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
!unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
!rm text8.zip source-archive.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 29.8M  100 29.8M    0     0   874k      0  0:00:35  0:00:35 --:--:--  871k
    Archive:  text8.zip
      inflating: text8                   
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  218k  100  218k    0     0   218k      0  0:00:01 --:--:--  0:00:01  888k



```python
! wget https://raw.githubusercontent.com/tensorflow/models/master/tutorials/embedding/word2vec_ops.cc
! wget https://raw.githubusercontent.com/tensorflow/models/master/tutorials/embedding/word2vec_kernels.cc

! TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
! TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
! g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

    --2018-07-03 01:29:24--  https://raw.githubusercontent.com/tensorflow/models/master/tutorials/embedding/word2vec_ops.cc
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.200.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.200.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2604 (2.5K) [text/plain]
    Saving to: 'word2vec_ops.cc'
    
    word2vec_ops.cc     100%[===================>]   2.54K  --.-KB/s    in 0s      
    
    2018-07-03 01:29:24 (14.5 MB/s) - 'word2vec_ops.cc' saved [2604/2604]
    
    --2018-07-03 01:29:24--  https://raw.githubusercontent.com/tensorflow/models/master/tutorials/embedding/word2vec_kernels.cc
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.200.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.200.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 12859 (13K) [text/plain]
    Saving to: 'word2vec_kernels.cc'
    
    word2vec_kernels.cc 100%[===================>]  12.56K  --.-KB/s    in 0.03s   
    
    2018-07-03 01:29:25 (468 KB/s) - 'word2vec_kernels.cc' saved [12859/12859]
    
    /bin/sh: 1: Syntax error: "(" unexpected
    /bin/sh: 1: Syntax error: "(" unexpected
    /bin/sh: 1: Bad substitution



```python
import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-05452877a836> in <module>()
          9 import tensorflow as tf
         10 
    ---> 11 word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
    

    NameError: name '__file__' is not defined



```python
flags = tf.app.flags

flags.DEFINE_string("save_path", '/tmp/', "Directory to write the model.")
flags.DEFINE_string(
    "train_data", 'text8',
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", 'questions-words.txt', "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")

FLAGS = flags.FLAGS

```


```python
class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)


    # The text file for eval.
    self.eval_data = FLAGS.eval_data

```


```python
class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()

  def read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data. A text file.
    ## use class from 'word2vec_ops.so'
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                    batch_size=opts.batch_size,
                                                    window_size=opts.window_size,
                                                    min_count=opts.min_count,
                                                    subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train_word2vec(w_in,
                                          w_out,
                                          examples,
                                          labels,
                                          lr,
                                          vocab_count=opts.vocab_counts.tolist(),
                                          num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.global_step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in range(opts.vocab_size):
        vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
        f.write("%s %d\n" % (vocab_word,
                             opts.vocab_counts[i]))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    self.saver = tf.train.Saver()

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in range(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(5)  # Reports our progress once a while.
      (epoch, step, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    try:
      total = self._analogy_questions.shape[0]
    except AttributeError as e:
      raise AttributeError("Need to read analogy questions.")

    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        print(c)
        break
    print("unknown")

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))


```


```python

def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
      model.read_analogies() # Read analogy questions
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      model.eval()  # Eval analogies.
    # Perform a final save.
    model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.global_step)
 
if __name__ == "__main__":
  tf.app.run()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-593b0eb8ca86> in <module>()
         18 
         19 if __name__ == "__main__":
    ---> 20   tf.app.run()
    

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/platform/app.py in run(main, argv)
        123   # Call the main function, passing through any arguments
        124   # to the final program.
    --> 125   _sys.exit(main(argv))
        126 


    <ipython-input-15-593b0eb8ca86> in main(_)
          8   with tf.Graph().as_default(), tf.Session() as session:
          9     with tf.device("/cpu:0"):
    ---> 10       model = Word2Vec(opts, session)
         11       model.read_analogies() # Read analogy questions
         12     for _ in xrange(opts.epochs_to_train):


    <ipython-input-14-9e67045eace6> in __init__(self, options, session)
          7     self._word2id = {}
          8     self._id2word = []
    ----> 9     self.build_graph()
         10     self.build_eval_graph()
         11     self.save_vocab()


    <ipython-input-14-9e67045eace6> in build_graph(self)
         42     # The training data. A text file.
         43     (words, counts, words_per_epoch, current_epoch, total_words_processed,
    ---> 44      examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
         45                                                     batch_size=opts.batch_size,
         46                                                     window_size=opts.window_size,


    NameError: name 'word2vec' is not defined

