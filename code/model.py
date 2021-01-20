import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers

from utils import *

def get_session(gpu_fraction=0.5):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Model:

    def __init__(self):
        self.nn = None
        self.res = None

    def create_md(self, attributes):
        nn = models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(attributes, )),
            layers.BatchNormalization(),
            layers.Dropout(0.01),
            layers.Dense(768, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.01),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.01),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.01),
            layers.Dense(1, activation='sigmoid')
        ])

        nn.summary()
        nn.compile(optimizer='Adam',
                   loss='binary_crossentropy',
                   metrics=[self.tpr, self.fpr])

        return nn
        
    def tpr(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        return tp / (tp + fn + K.epsilon())

    def fpr(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        return fp / (fp + tn + K.epsilon())

    def predict(self, x, attributes=207):
        if self.nn == None :
            if not os.path.isfile("../param/model.h5") :
                print("Do training first!")
                return 0

            self.nn = self.create_md(attributes)
            self.nn.load_weights("../param/model.h5")

        y_pred = self.nn.predict_classes(x, batch_size=500)
        return y_pred

    def train_md(self, x_train, x_test, y_train, y_test):
        history_callback = self.nn.fit(x = x_train,
                                       y = y_train,
                                       epochs = 100,
                                       validation_data=(x_test, y_test),
                                       batch_size=500)

        t_tpr_history = history_callback.history["tpr"]
        t_fpr_history = history_callback.history["fpr"]
        v_tpr_history = history_callback.history["val_tpr"]
        v_fpr_history = history_callback.history["val_fpr"]

        return t_tpr_history, t_fpr_history, v_tpr_history, v_fpr_history

    def train(self, x, y, run_cv=False, attributes=207):

        if run_cv == True :
            n_fold = 10
            kfold = KFold(n_splits=n_fold, shuffle=True, random_state=9449)
        
            itr = 0
            res = {'t_tprs':[], 't_fprs':[], 'v_tprs':[], 'v_fprs':[]}
        
            for train_index, test_index in kfold.split(y) :
                itr = itr + 1
                print("Iteration : ", itr)
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.nn = self.create_md(attributes)

                t_tpr, t_fpr, v_tpr, v_fpr = self.train_md(x_train, x_test, y_train, y_test)

                res['t_tprs'].append(t_tpr)
                res['t_fprs'].append(t_fpr)
                res['v_tprs'].append(v_tpr)
                res['v_fprs'].append(v_fpr)
                
            self.res = {'t_tprs':np.mean(res['t_tprs'], axis=0),
                        't_fprs':np.mean(res['t_fprs'], axis=0),
                        'v_tprs':np.mean(res['v_tprs'], axis=0),
                        'v_fprs':np.mean(res['v_fprs'], axis=0)}

        else :
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

            self.nn = self.create_md(attributes)

            t_tpr, t_fpr, v_tpr, v_fpr = self.train_md(x_train, x_test, y_train, y_test)

            self.res = {'t_tprs':t_tpr, 't_fprs':t_fpr, 'v_tprs':v_tpr, 'v_fprs':v_fpr}

        print("------------Final results------------")
        print("Training    : TPR (%.3f) FPR (%.3f)"%(self.res['t_tprs'][-1], self.res['t_fprs'][-1]))
        print("Validation  : TPR (%.3f) FPR (%.3f)"%(self.res['v_tprs'][-1], self.res['v_fprs'][-1]))

        self.nn.save("../param/model.h5")




