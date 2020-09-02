# -*- coding: utf-8 -*-
"""Copy of bestTrain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O21we1WayQzEDE6PZVNhlltFoLcvkkfk
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
import pandas as pd
import numpy as np
import pickle
import copy
import argparse 
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf #machine learningt
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
from google.colab import drive
drive.mount('/content/drive')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, metavar='',help='(initial learning rate for gradient descent based algorithms)',default = .001)
parser.add_argument('--init', type = int , metavar ='', help = '(weghit initialization method)', default = 1)
parser.add_argument( '--batch_size',type=int,metavar='', help='(the batch size to be used)', default= 128)
parser.add_argument('--epochs',type=int,metavar='', help='(number of passes over the data)',default = 1)
parser.add_argument('--dropout_prob',type=float,metavar='', help='(droput)',default = .5)
parser.add_argument('--decode_method',type=int,metavar='', help='(droput)',default = 0)
parser.add_argument('--beam_width',type=int,metavar='', help='(droput)',default = 1)
parser.add_argument( '--save_dir',type=str, metavar='',help='(the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network)', default='/content/drive/My Drive/deeplearning/checkpoints/dev')
parser.add_argument("--train",type=str,help="path to training dataset",default="/content/drive/My Drive/deeplearning/train.csv")
parser.add_argument("--val",type=str,help="path to validation dataset",default="/content/drive/My Drive/deeplearning/valid.csv")
args = parser.parse_args()


learning_rate =args.lr
batch_size = args.batch_size
initialization = args.init
epochs = args.epochs
train = args.train
dropout = args.dropout_prob
valid = args.val
directory= args.save_dir
decode_method = args.decode_method
beam_width = args.beam_width
training_iters = epochs
learning_rate = learning_rate
# learning_rate =args.lr
batch_size = batch_size


display_step = 100

epochs = epochs
batch_size = batch_size

rnn_size = 512
num_layers = 2

encoding_embedding_size = 256
decoding_embedding_size = 256

learning_rate = learning_rate
keep_probability = 1-dropout

train = pd.read_csv(train)
valid = pd.read_csv(valid)
test = pd.read_csv('/content/drive/My Drive/deeplearning/test_final.csv')

data=train
X=data
X=X.loc[:,X.columns !='id']
X = X['HIN']


data=train
Z=data
Z=Z.loc[:,Z.columns !='id']
Z = Z['ENG']

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def create_lookup_tables(X):
    # make a list of unique words
  
    vocab=set()
    for i in range(len(X)):
      t=set(X[i])-set(" ")
      vocab=vocab.union(t)
   
    
    # (1)
    # starts with the special tokens
    vocab_to_int = copy.copy(CODES)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    # (2)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
        1st, 2nd args: raw string text to be converted
        3rd, 4th args: lookup tables for 1st and 2nd args respectively
    
        return: A tuple of lists (source_id_text, target_id_text) converted
    """
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []
    
    # make a list of sentences (extraction)
    source_sentences = source_text
    target_sentences = target_text
    
    
    
    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id

def preprocess_and_save_data(source_path, target_path, text_to_ids):
    # Preprocess
    
    # load original data (English, French)
    source_text = source_path
    target_text = target_path

    # create lookup tables for English and French data
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('/content/drive/My Drive/deeplearning/preprocess.p', 'wb'))

preprocess_and_save_data(Z, X, text_to_ids)

import pickle

def load_preprocess():
    with open('/content/drive/My Drive/deeplearning/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)

import numpy as np

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()

def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len

def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, 
                                       embed, 
                                       dtype=tf.float32)
    ((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cells,
                                    cell_bw=stacked_cells,
                                    inputs=embed,
                                    
                                    dtype=tf.float32)
    )


#Concatenates tensors along one dimension.
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)


    return encoder_outputs, encoder_fw_final_state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    
    # attention_states: [batch_size, max_time, num_units]
 
    """
    Create a training process in decoding layer 
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs,final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs, final_state

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer 
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, infer_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs,infer_final_state

def decoding_layer(dec_input, encoder_state,enc_output,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
#     output_layer = Dense(target_vocab_size,
#            kernel_initializer = tf.truncated_normal_initializer(  
#                                     mean=0.0, 
#                                     stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                      rnn_size,
                      enc_output,
                      target_sequence_length,
                      normalize=False,
                      name='BahdanauAttention')
    
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(cells,
                                                          attn_mech,
                                                          rnn_size,True)

    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=encoder_state)

    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output, final_state = decoding_layer_train(
                                            initial_state,
                                            dec_cell, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output ,infer_final_state = decoding_layer_infer(initial_state, 
                                            dec_cell, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output,final_state,infer_final_state)

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output,final_state,infer_final_state = decoding_layer(dec_input,
                                               enc_states, 
                                               enc_outputs,
                                               target_sequence_length, 
                                               max_target_sentence_length,
                                               rnn_size,
                                              num_layers,
                                              target_vocab_to_int,
                                              target_vocab_size,
                                              batch_size,
                                              keep_prob,
                                              dec_embedding_size)
    
    return train_output, infer_output,final_state,infer_final_state

save_path = '/content/drive/My Drive/deeplearning/checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()
    
    train_logits, inference_logits,final_state,infer_final_state = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                          target_vocab_to_int['<PAD>']))  
import matplotlib.pyplot as plt 
def plot_attention(attention_map, input_tags = None, output_tags = None):    
    attn_len = len(attention_map)

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # Add labels
    ax.set_yticks(range(attn_len))
    if output_tags != None:
      ax.set_yticklabels(output_tags[:attn_len])

    ax.set_xticks(range(attn_len))
    if input_tags != None:
      ax.set_xticklabels(input_tags[:attn_len], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    plt.show()
    plt.savefig('/content/drive/My Drive/deeplearning/attention.png')

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    peak_loss = 1000
    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})

            alignments = sess.run(final_state.alignment_history.stack(),{input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})
            alignments1 = sess.run(infer_final_state.alignment_history.stack(),{input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})
            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
          if epoch_i % 5 == 0:
            if(loss >peak_loss):
              break
            else:
              peak_loss = loss
              
               
               
            
    
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

plot_attention(alignments[:, 78, :])
plot_attention(alignments1[:, 78, :])
def save_params(params):
    with open('/content/drive/My Drive/deeplearning/params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('/content/drive/My Drive/deeplearning/params.p', mode='rb') as in_file:
        return pickle.load(in_file)

save_params(save_path)

import tensorflow as tf
import numpy as np


_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()
load_path = load_params()

def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results
import csv  
fro = test['ENG']
pro = fro.values
transformed = []
for i in pro:
  
  sen = sentence_to_seq(i, source_vocab_to_int)
  transformed.append(sen)

translate_sentence = 'S H Y A M'

translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
predicted_logits = []
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    for translate_sentence in transformed:

      translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         keep_prob: 1.0})[0]
     
      predicted_logits.append(translate_logits)
      
new_logits = []
for q in  predicted_logits:

  yo = format(" ".join([target_int_to_vocab[i] for i in q]))
  yo = yo.replace("<EOS>", "")
  print(yo)
  new_logits.append(yo)
final = []
count =0
for lo in new_logits:
  xo = []
  xo.append(count)
  xo.append(lo)
  final.append(xo)
  count = count +1
  
print("fianl",final)


  
  
  
  
with open("/content/drive/My Drive/deeplearning/submission.csv", 'w', newline='') as f:
        thewriter = csv.writer(f)
        row = len(final)
        thewriter.writerow(['id', 'HIN'])
        for i in range(row):
            thewriter.writerow(final[i])


# print('Input')
# print('  Word Ids:      {}'.format([i for i in translate_sentence]))
# print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

# print('\nPrediction')
# print('  Word Ids:      {}'.format([i for i in translate_logits]))
# print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

hindi,_= create_lookup_tables(X)
print(hindi)

eng,_ = create_lookup_tables(Z)
print(eng)

f,o= text_to_ids(Z, X, eng, hindi)
print(f[0])

data=train
X=data
X=X.loc[:,X.columns !='id']
#print(X)
print(X.shape)
demo=set()
for i in range(len(X)):
    t=set(X['ENG'][i])-set(" ")
    demo=demo.union(t)
    
print(demo)
y = X['ENG']
y=y.values
p= list(y[0])
print(p)

!pip install pydrive

print(tf.__version__)