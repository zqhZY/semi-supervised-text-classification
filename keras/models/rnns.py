from keras.engine import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout


def get_lstm(args):
    embedding_dim = args.rnn_embed_dim
    recurrent_dropout = args.recurrent_dropout
    num_lstm = args.rnn_num_lstm
    max_seq_len = args.rnn_max_seq_len
    drop = args.dropout
    num_dense = args.rnn_num_dense

    if args.embedding_matrix != None:
        embedding_layer = Embedding(args.nb_words,
                                    embedding_dim,
                                    weights=[args.embedding_matrix],
                                    input_length=max_seq_len,
                                    trainable=False)
    else:
        embedding_layer = Embedding(args.nb_words, embedding_dim, input_length=max_seq_len)
    lstm_layer = LSTM(num_lstm, dropout=drop, recurrent_dropout=recurrent_dropout)

    inputs = Input(shape=(max_seq_len,), dtype='int32')
    embedding_seq = embedding_layer(inputs)
    features = lstm_layer(embedding_seq)

    dense = Dense(num_dense, activation="relu")(features)
    dropout = Dropout(drop)(dense)
    output = Dense(output_dim=5, activation='softmax')(dropout)
    # this creates a model that includes
    model = Model(input=inputs, output=output)
    print model.summary()

    return model




