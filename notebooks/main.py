#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import numpy as np
import tensorflow as tf
import training
# from aspp_models import fpn as get_model
from sklearn.model_selection import KFold
import random
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



# In[ ]:


path = "../data/video/study04"
result_path="../results"
if not os.path.exists(result_path):
    os.makedirs(result_path)
for i in range(1,6):
    if not os.path.exists(os.path.join(result_path,"study0"+str(i))):
        os.makedirs(os.path.join(result_path,"study0"+str(i)))
save_path = "../results/study04"
batch_size = 10
epochs = 20

#
# c0 = []
# c1 = []
# c2 = []
# c3 = []
# for d in os.listdir(path):
#     for r2,d2,f2 in os.walk(os.path.join(path,d)):
#         for file in f2:
#             filepath=os.path.join(r2,file)
#             if file[-5] == '1':
#                 c0.append(filepath)
#             if file[-5] == '2':
#                 c1.append(filepath)
#             if file[-5] == '3':
#                 c2.append(filepath)
#             if file[-5] == '4':
#                 c3.append(filepath)



# In[ ]:


# print(len(c0),len(c1),len(c2),len(c3))


# In[ ]:


# bound=max(max(len(c0),len(c1)),max(len(c2),len(c3)))


# In[ ]:

#
# c1_dups=random.sample(range(0, len(c1)), bound-len(c1))
# c2_dups=random.sample(range(0, len(c2)), bound-len(c2))
# c3_dups=random.sample(range(0, len(c3)), bound-len(c3))
# In[ ]:




# In[ ]:


# for i in c1_dups:
#     c1.append(c1[i])
# for i in c2_dups:
#     c2.append(c2[i])
# for i in c3_dups:
#     c3.append(c3[i])


# In[ ]:


# print(len(c0),len(c1),len(c2),len(c3))


# In[ ]:


# video=np.load(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\video\study01\p01\0_1.npy")


# In[ ]:


# video.shape


# In[ ]:




def data_generator(files, batchsize):
    while True:
        cnt = 0
        for i in range(len(files)):
            # print(files[i])
            image = np.load(files[i],encoding='latin1', allow_pickle=True)
            label = int(files[i][-5])-1

            if i == 0 or cnt == 0:
                xtrain = image
                ytrain = np.array([label])
            else:
                xtrain = np.concatenate((xtrain, image), axis=0)
                ytrain = np.concatenate((ytrain, np.array([label])), axis=0)
            cnt += 1
            if cnt == batchsize or i == (len(files)-1):
                cnt = 0
#                 print(xtrain.shape)
                yield xtrain, ytrain
        


# In[ ]:


import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Concatenate
from tensorflow.keras.layers import Input, TimeDistributed, Reshape, MaxPooling1D, Permute, Conv1D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import LearningRateScheduler


class WeightClip(Constraint):
    def __init__(self, c):
        self.c = c

    def __call__(self, p):
        return tf.keras.backend.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}



def fpn():
    input_layer = Input(shape=(112, 32, 32, 3))
    c1 = TimeDistributed(Conv2D(name="C1", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1)), name='TC1')(input_layer)
    c1 = TimeDistributed(BatchNormalization(name="B1"), name='TB1')(c1)

    c2 = TimeDistributed(Conv2D(name="C2", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1)), name='TC2')(c1)
    c2 = TimeDistributed(BatchNormalization(name="B2"), name='TB2')(c2)

    c3 = TimeDistributed(Conv2D(name="C3", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1)), name='TC3')(c2)
    c3 = TimeDistributed(BatchNormalization(name="B3"), name='TB3')(c3)

    c41 = TimeDistributed(Conv2D(name="C4", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=1), name='TC41')(c3)
    c42 = TimeDistributed(Conv2D(name="C4", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=2), name='TC42')(c3)
    c44 = TimeDistributed(Conv2D(name="C4", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=4), name='TC44')(c3)
    c46 = TimeDistributed(Conv2D(name="C4", filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=6), name='TC46')(c3)
    c4_conc = Concatenate(name='aspp1')([c41, c42, c44, c46])
    c4 = TimeDistributed((Conv2D(name='c4', filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1))), name='TC4')(c4_conc)

    m1 = TimeDistributed(MaxPooling2D(name='M1', pool_size=(2, 2), strides=(2, 2)), name='TM1')(c4)
    m1 = TimeDistributed(BatchNormalization(name='B4'), name='TB4')(m1)

    c5 = TimeDistributed(Conv2D(name='C5', filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1)), name='TC5')(m1)
    c5 = TimeDistributed(BatchNormalization(name="B5"), name='TB5')(c5)

    c61 = TimeDistributed(Conv2D(name='C6', filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=1), name='TC61')(c5)
    c62 = TimeDistributed(Conv2D(name='C6', filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=2), name='TC62')(c5)
    c64 = TimeDistributed(Conv2D(name='C6', filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=4), name='TC64')(c5)
    c66 = TimeDistributed(Conv2D(name='C6', filters=64, kernel_size=3, activation='relu', padding='same', dilation_rate=6), name='TC66')(c5)
    c6_conc = Concatenate(name='aspp2')([c61, c62, c64, c66])
    c6 = TimeDistributed((Conv2D(name='c4', filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1))), name='TC6')(c6_conc)

    m2 = TimeDistributed(MaxPooling2D(name='M2', pool_size=(2, 2), strides=(2, 2)), name='TM2')(c6)
    m2 = TimeDistributed(BatchNormalization(name="B6"), name='TB6')(m2)

    c71 = TimeDistributed(Conv2D(name="C7", filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=1), name='TC71')(m2)
    c72 = TimeDistributed(Conv2D(name="C7", filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=2), name='TC72')(m2)
    c74 = TimeDistributed(Conv2D(name="C7", filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=4), name='TC74')(m2)
    c76 = TimeDistributed(Conv2D(name="C7", filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1), dilation_rate=6), name='TC76')(m2)
    c7_conc = Concatenate(name='aspp3')([c71, c72, c74, c76])
    c7 = TimeDistributed(Conv2D(name='c7', filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.1)), name='Tc7')(c7_conc)



    uc7 = TimeDistributed(UpSampling2D(name="UpSamp-C7", size=(2, 2), interpolation='bilinear'), name='TD-UP-C7')(c7)
    oc6 = TimeDistributed(Conv2D(name='1D-C6', filters=128, kernel_size=1, activation='relu', padding='same'), name='TD-1D-C6')(c6)
    fpn1 = Concatenate(name="FPN1")([uc7, oc6])
    c100 = TimeDistributed(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'), name="FPN1_CNN")(fpn1)
    m100 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name="FPN1_MP")(c100)
    m100 = TimeDistributed(Flatten(), name="Flatten_FPN1_M100")(m100)
    lstm100 = LSTM(128, activation='tanh', kernel_constraint=WeightClip(100), name="LSTM100")(m100)
    conv1d100 = Conv1D(name="FPN1_CONV1D", filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(m100)
    conv1d100 = Flatten(name="Flatten_FPN1_C1D")(conv1d100)
    conc100 = Concatenate(name="CONC_FPN1")([lstm100, conv1d100])
    conc100 = Dropout(0.2)(conc100)
    d100 = Dense(1024, activation='relu')(conc100)

    ufpn1 = TimeDistributed(UpSampling2D(name="UpSamp-FPN1", size=(2, 2), interpolation='bilinear'), name='TD-UP-FPN1')(fpn1)
    oc4 = TimeDistributed(Conv2D(name='1D-C4', filters=256, kernel_size=1, activation='relu', padding='same'), name='TD-1D-C4')(c4)
    fpn2 = Concatenate(name='FPN2')([ufpn1, oc4])
    c200 = TimeDistributed(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'), name="FPN2_CNN")(fpn2)
    m200 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name="FPN2_MP")(c200)
    m200 = TimeDistributed(Flatten(), name="Flatten_FPN2_M200")(m200)
    lstm200 = LSTM(128, activation='tanh', kernel_constraint=WeightClip(100), name="LSTM200")(m200)
    conv1d200 = Conv1D(name="FPN2_CONV1D", filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(m200)
    conv1d200 = Flatten(name="Flatten_FPN2_C1D")(conv1d200)
    conc200 = Concatenate(name="CONC_FPN2")([lstm200, conv1d200])
    conc200 = Dropout(0.2)(conc200)
    d200 = Dense(1024, activation='relu')(conc200)

    m3 = TimeDistributed(MaxPooling2D(name='M3', pool_size=(2, 2), strides=(2, 2)), name='TM3')(c7)
    m3 = TimeDistributed(BatchNormalization(name="B7"), name='TB7')(m3)
    m3 = TimeDistributed(Flatten(), name='TD-Flatten1')(m3)

    #mp_conc = Concatenate(name="MP_CONC")([m3, m100, m200])

    lstm1 = LSTM(128, activation='tanh', kernel_regularizer=l2(0.1), kernel_constraint=WeightClip(100), name="L1")(m3)
    c8 = Conv1D(name="1DCONV", filters=64, kernel_size=3, strides=1, kernel_regularizer=l2(0.1), activation='relu', padding='valid')(m3)
    c8 = Flatten()(c8)
    conc1 = Concatenate(name="CONC_LSTM_1D")([lstm1, c8])
    conc1 = Dropout(0.2)(conc1)
    conc2 = Concatenate(name='conc')([conc1, d200, d100])
    d1 = Dense(256, activation='relu')(conc2)
    #d1 = Dense(256, activation='relu')(d1)
    d1 = Dropout(0.2)(d1)
    d1 = Dense(4, activation='softmax')(d1)
    mod = Model(inputs=input_layer, outputs=d1)
    adam = Adam(learning_rate=0.00001)
    mod.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     print("yes")
    return mod


# In[ ]:

path_pre = "../data/video"
save_path_pre = "../results"
studys=[
    # "study02"
    # "study02"
    # "study03"
    "study04"
    # "study05"
]
for study in studys:
    test_acc = []
    i = 1
    for r,participants,f in os.walk(os.path.join(path_pre,study)):
        test_lis=[]
        train_lis=[]
        for part in participants:
            for r1,d,f1 in os.walk(os.path.join(r,part)):
                for filenames in f1:
                    test_lis.append(os.path.join(r1,filenames))
            for parts in participants:
                if parts != part:
                    for r1,d,ft in os.walk(os.path.join(r,parts)):
                        for filenames in ft:
                            train_lis.append(os.path.join(r1,filenames))
            random.seed(1)
            random.shuffle(train_lis)
            random.shuffle(train_lis)
            random.seed(1)
            random.shuffle(test_lis)
            random.shuffle(test_lis)
            train_gen = data_generator(files=train_lis, batchsize=batch_size)
            print("[INFO] Training Model...")
            model= fpn()
            history = model.fit(train_gen, verbose=1, epochs=epochs, steps_per_epoch=len(train_lis) // batch_size)
            print("[INFO] Model trained. Getting predictions.")
            test_accuracy = training.get_predictions(model, test_lis, i, save_path, batch_size)
            i += 1
            test_acc.append(test_accuracy)
            del model
            tf.keras.backend.clear_session()
    print("----------------------------------------------------")
    print("Average test accuracy: ", np.sum(test_acc) / len(test_acc))
    print(test_acc)
    print("[DONE]...")
#     for trainidx, testidx in kf.split(folds):
# #     print(trainidx, testidx)
#         train_lis = list(np.array(c0)[trainidx]) + list(np.array(c1)[trainidx]) + list(np.array(c2)[trainidx]) + list(np.array(c3)[trainidx])
#         random.seed(1)
#         random.shuffle(train_lis)
#         random.shuffle(train_lis)
#         test_lis = list(np.array(c0)[testidx]) + list(np.array(c1)[testidx]) + list(np.array(c2)[testidx]) + list(np.array(c3)[testidx])
#         random.seed(1)
#         random.shuffle(test_lis)
#         random.shuffle(test_lis)
#
#         train_gen = data_generator(files=train_lis, batchsize=batch_size)
# #     print(train_gen)
#         print("[INFO] Training Model...")
#     # strategy = tf.distribute.MirroredStrategy()
#     # with strategy.scope():
#         model = fpn()
#         history = model.fit(train_gen, verbose=1, epochs=epochs, steps_per_epoch=len(train_lis)//batch_size)
#
#         print("[INFO] Model trained. Getting predictions.")
#         test_accuracy = training.get_predictions(model, test_lis, i, save_path, batch_size)
#         i += 1
#         test_acc.append(test_accuracy)
#         del model
#         tf.keras.backend.clear_session()
#
#     print("----------------------------------------------------")
#     print("Average test accuracy: ", np.sum(test_acc)/len(test_acc))
#     print(test_acc)
#     print("[DONE]...")

