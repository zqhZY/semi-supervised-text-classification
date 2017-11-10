# -*- coding:utf-8 -*-
import sys
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

reload(sys)
sys.setdefaultencoding("utf-8")


def self_train(x, y, unlabel_set, model, args):

    X_train, X_dev, y_train, y_dev = train_test_split(x, y, test_size=0.06, random_state=42)
    while len(unlabel_set) != 0:
        print "---------------------------------------new round----------------------------------------"
        # train model, early stop
        checkpoint = ModelCheckpoint(args.save_dir + '/self_train_weights_latest.hdf5', monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True, mode='auto')
        early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='auto')
        adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                  callbacks=[early_stop, checkpoint, TensorBoard(log_dir='./log_dir')],
                  validation_data=(X_dev, y_dev))  # starts training

        # test for unlabel_set
        model.load_weights(args.save_dir + '/self_train_weights_latest.hdf5')
        predict_probs = model.predict(unlabel_set)
        predict_labels = np.argmax(predict_probs, axis=1)
        predict_labels = to_categorical(predict_labels, num_classes=14)

        # add predicted sample (prob >= threshold) to X_trian , rm these samples from unlabeled data
        indexs_toadd = np.where(np.max(predict_probs, axis=1) >= args.threshold)
        indexs_stayed = np.where(np.max(predict_probs, axis=1) < args.threshold)
        X_train = np.append(X_train, unlabel_set[indexs_toadd], axis=0)
        y_train = np.append(y_train, predict_labels[indexs_toadd], axis=0)
        tmp_unlabel_set = unlabel_set[indexs_stayed]

        # loop until unlabel_set.len = 0 or indexs_toadd.len = 0 (confidence >= threshold)
        if len(tmp_unlabel_set) != len(unlabel_set):
            unlabel_set = tmp_unlabel_set
            print "now unlabel set len: ", len(unlabel_set)
        else:
            # no prob > threadhold break loop
            print "no confidence unlabel data, break..."
            break
