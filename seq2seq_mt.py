import tensorflow as tf
import numpy as np
import time
from sys import argv

begin = time.time()

# take in path names to various datasets
ftr_path = argv[1]
etr_path = argv[2]
fte_path = argv[3]
ete_path = argv[4]

# set hyperparameters
window_sz = 13
batch_sz = 20
embed_sz = 30
state_sz = 64

# Define a procedure to read in the datasets and produce
# resulting vocabulary, numeric corpus, and sentence length log
def pre_process(file_path, vocab):
	# Read in the file, split by line and white space
	with open(file_path) as f:
		corp = [line.split() for line in f.readlines()]

	# First vocab term is the stop token
	vocab['*STOP*'] = 0

	# intialize corpus, sentence length log, find initial index value
	num_corp = []
	lengths = []
	i = max(vocab.values()) + 1

	for sent in corp:
		# create a new sentence
		s = []
		for word in sent:
			# if the word is not in vocab yet add it
			if word not in vocab:
				vocab[word] = i
				i += 1
			# append word index to current sentence
			s.append(vocab[word])

		# pad the sentence
		s = s + [0]*(window_sz-len(s))
		# add current sentence to corpus
		num_corp.append(s)
		# add current sentence length to length log
		lengths.append(len(sent)+1)

	# return corpus, vocab, and length log
	return np.array(num_corp), vocab, lengths

# pre-process the four corpus, make sure vocabularies encompass both
# training and test vocab
fr_tr, fr_vocab, ftr_len = pre_process(ftr_path, {})
fr_te, fr_vocab, fte_len = pre_process(fte_path, fr_vocab)
en_tr, en_vocab, etr_len = pre_process(etr_path, {})
en_te, en_vocab, ete_len = pre_process(ete_path, en_vocab)

# keep track of the vocab sizes and corpus lengths
fv_sz = len(fr_vocab)
ev_sz = len(en_vocab)
fctr_sz = len(fr_tr)
fcte_sz = len(fr_te)
ectr_sz = len(en_tr)
ecte_sz = len(en_te)


# initialize placeholders
# Encoder input represents incoming french
enc_in = tf.placeholder(dtype=tf.int32, shape=[batch_sz,window_sz])
# Decoder input represents modified incoming english
dec_in = tf.placeholder(dtype=tf.int32, shape=[batch_sz,window_sz])
# Answer represents unmodified english
ans = tf.placeholder(dtype=tf.int32, shape=[batch_sz,window_sz])
# sentence lengths are the lengths of the corresponding english sentences
sent_len = tf.placeholder(dtype=tf.int32, shape=[batch_sz])
# produce a mask from the sentence lengths
mask = tf.sequence_mask(sent_len, maxlen=window_sz, dtype=tf.float32)

# Define the encoder variables and process
with tf.variable_scope('enc'):
	# Initialize French embeddings
	F = tf.Variable(tf.random_normal([fv_sz, embed_sz], stddev=.1))
	# grab embeddings for input batch
	embed = tf.nn.embedding_lookup(F, enc_in)
	# initialize the encoder, feed in the embeddings to the GRU
	encoder = tf.contrib.rnn.GRUCell(state_sz)
	initial = encoder.zero_state(batch_sz, tf.float32)
	enc_out, enc_state = tf.nn.dynamic_rnn(encoder, embed, initial_state=initial)

# Decoder varibles and process
with tf.variable_scope('dec'):
	# Initialize English embeddings
	E = tf.Variable(tf.random_normal([ev_sz, embed_sz], stddev=.1))
	# get batch embeddings
	embed = tf.nn.embedding_lookup(E, dec_in)
	# initialize the decoder, feed in the corresponding embeddings and state
	# from the encoder
	decoder = tf.contrib.rnn.GRUCell(state_sz)
	dec_out, _ = tf.nn.dynamic_rnn(decoder, embed, initial_state=enc_state)

# initialize the linear transformation to return output to vocab size
W = tf.Variable(tf.random_normal([state_sz, ev_sz], stddev=.1))
b = tf.Variable(tf.random_normal([ev_sz], stddev=.1))

# produce the logits and loss
logits = tf.tensordot(dec_out, W, axes=[[2],[0]]) + b
loss = tf.contrib.seq2seq.sequence_loss(logits, ans, mask)

# Predict word with highest logit value
pred = tf.argmax(logits, axis=2)

# minimize loss
tr = tf.train.AdamOptimizer(.005).minimize(loss)

ses = tf.Session()
ses.run(tf.global_variables_initializer())


# number of batches for train and test
num_tr_batches = fctr_sz/batch_sz
num_te_batches = fcte_sz/batch_sz

tr_loss = 0.0
for i in range(num_tr_batches):
	# Print average loss every 100 iterations
	if i % 1000 == 0:
		print 'Iteration', i
		print 'Train Loss:', tr_loss/1000
		# reset
		tr_loss = 0.0
		print

	# Find starting and ending index for batch
	start = i*batch_sz
	end = start + batch_sz

	# grab the batches for french and english. Modify english for decoder input
	french = fr_tr[start:end,:]
	english = en_tr[start:end,:]
	m_english = np.concatenate((np.zeros((batch_sz,1)),english[:,:window_sz-1]),axis=1)

	# get sentence lengths
	lens = etr_len[start:end]

	# Run the session. Keep track of loss.
	l, _ = ses.run([loss, tr], feed_dict={enc_in:french, dec_in:m_english, ans:english, sent_len:lens})
	tr_loss += l


correct = 0.0
total = 0.0
for j in range(num_te_batches):
	# grab batch indicies
	start = j*batch_sz
	end = start + batch_sz

	# grab batches, modify english for decoder input
	french = fr_te[start:end,:]
	english = en_te[start:end,:]
	m_english = np.concatenate((np.zeros((batch_sz,1)),english[:,:window_sz-1]),axis=1)

	# get sentence lengths
	lens = ete_len[start:end]

	# run session. Get prediction and mask
	prediction, m = ses.run([pred, mask], feed_dict={enc_in:french, dec_in:m_english, ans:english, sent_len:lens})

	# accumulate number correct. Multiply by mask to zero out values we don't want to count
	correct += np.sum(np.equal(english,prediction) * m)
	# accumulate mask sums to get total words in actual sentence length per batch
	total += np.sum(m)

# calculate accuracy
accuracy = correct/total

# print runtime and accuracy
print 'Runtime:'
print time.time()-begin
print
print 'Accuracy:'
print accuracy





