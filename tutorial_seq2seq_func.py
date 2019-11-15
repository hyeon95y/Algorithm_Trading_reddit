import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, BatchNormalization

# from [Natural Language Understanding with Sequence to Sequence Models](https://towardsdatascience.com/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b)

def max_length(tensor) : 
    return max(len(t) for t in tensor)

def create_tensors(input_tensor, target_tensor, nb_sample=9999999, max_len=0) :
    len_input, len_target = max_length(input_tensor), max_length(target_tensor)
    len_input = max(len_input, max_len)
    len_target = max(len_target, max_len)
    
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                              maxlen=len_input,
                                                              padding='post')
    teacher_data = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                maxlen=len_target,
                                                                padding='post')
    
    target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
    target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
    target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))
    
    nb = len(input_data)
    p = np.random.permutation(nb)
    input_data = input_data[p]
    teacher_data = teacher_data[p]
    target_data = target_data[p]

    return input_data[:min(nb_sample, nb)], teacher_data[:min(nb_sample, nb)], target_data[:min(nb_sample, nb)], len_input, len_target 

def get_vocab_size(t2i_train, t2i_test, s2i_train, s2i_test) :
    vocab_in_size = len({**t2i_train, **t2i_test})
    vocab_out_size = len({**s2i_train, **s2i_test})
    return vocab_in_size, vocab_out_size

def create_model(input_data_train, len_input_train, vocab_in_size, vocab_out_size, BATCH_SIZE = 64) : 
    BUFFER_SIZE = len(input_data_train)
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    
    # encoder
    encoder_inputs = Input(shape=(len_input_train,))
    encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
    encoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb(encoder_inputs))
    encoder_states = [state_h, state_c]
    
    # decoder
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
    decoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
    decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)
    
    decoder_d1 = Dense(units, activation='relu')
    decoder_d2 = Dense(vocab_out_size, activation='softmax')
    
    #decoder_out = decoder_d2(Dropout(rate=.4)(decoder_d1(Dropout(rate=.4)(decoder_lstm_out))))
    decoder_out = decoder_d2(BatchNormalization()(decoder_d1(BatchNormalization()(decoder_lstm_out))))
    
    model = Model([encoder_inputs, decoder_inputs], decoder_out)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    return model