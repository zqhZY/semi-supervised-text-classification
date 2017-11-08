# -*- coding:utf-8 -*-

from keras.layers import Input,Activation,  Dense, Embedding, merge, Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.layers.core import Reshape, Flatten
from keras.models import Model
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


def get_cnn(args):
    # this returns a tensor
    embedding_dim = args.embed_dim
    filter_sizes = args.filter_sizes
    num_filters = args.kernel_num
    max_seq_len = args.max_seq_len
    drop = args.dropout

    inputs = Input(shape=(args.max_seq_len,), dtype='int32')
    if args.embedding_matrix != None:
        embedding = Embedding(output_dim=embedding_dim, input_dim=args.nb_words, input_length=max_seq_len, weights=[args.embedding_matrix], trainable=True)(inputs)
    else:
        embedding = Embedding(output_dim=embedding_dim, input_dim=args.nb_words, input_length=max_seq_len)(inputs)

    reshape = Reshape((args.max_seq_len,embedding_dim,1))(embedding)

    conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', dim_ordering='tf')(reshape)
    conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal',  dim_ordering='tf')(reshape)
    conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', dim_ordering='tf')(reshape)
    conv_3 = Convolution2D(num_filters, filter_sizes[3], embedding_dim, border_mode='valid', init='normal', dim_ordering='tf')(reshape)

    conv_0 = BatchNormalization()(conv_0)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_3 = BatchNormalization()(conv_3)


    conv_0 = Activation("relu")(conv_0)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Activation("relu")(conv_2)
    conv_3 = Activation("relu")(conv_3)

    #conv_0 = Dropout(drop)(conv_0)
    #conv_1 = Dropout(drop)(conv_1)
    #conv_2 = Dropout(drop)(conv_2)
    #conv_3 = Dropout(drop)(conv_3)

    maxpool_0 = MaxPooling2D(pool_size=(max_seq_len - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(max_seq_len - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(max_seq_len - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)
    maxpool_3 = MaxPooling2D(pool_size=(max_seq_len - filter_sizes[3] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_3)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3*num_filters,))(merged_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(output_dim=5, activation='softmax')(dropout)
    # this creates a model that includes
    model = Model(input=inputs, output=output)
    print model.summary()


    return model
