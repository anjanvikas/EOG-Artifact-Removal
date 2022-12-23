import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import numpy as np
import tensorflow as tf
import training
from models import fpn as get_model
from sklearn.model_selection import KFold
import random

path = '../../../processed_data'
save_path = '../results/'
batch_size = 8
epochs = 75

files = os.listdir(path)
files = [os.path.join(path, file) for file in files]
c0 = [file for file in files if file[-5] == '0']
c1 = [file for file in files if file[-5] == '1']
folds = np.arange(1, len(c0)+1, 1)
kf = KFold(n_splits=10, shuffle=True, random_state=1)
test_acc = []
i = 1
for trainidx, testidx in kf.split(folds):
    train_lis = list(np.array(c0)[trainidx]) + list(np.array(c1)[trainidx])
    random.seed(1)
    random.shuffle(train_lis)
    random.shuffle(train_lis)
    test_lis = list(np.array(c0)[testidx]) + list(np.array(c1)[testidx])
    random.seed(1)
    random.shuffle(test_lis)
    random.shuffle(test_lis)

    train_gen = training.data_generator(files=train_lis, batchsize=batch_size)

    print("[INFO] Training Model...")
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = get_model()
    history = model.fit(train_gen, verbose=1, epochs=epochs, steps_per_epoch=len(train_lis)//batch_size)

    print("[INFO] Model trained. Getting predictions.")
    test_accuracy = training.get_predictions(model, test_lis, i, save_path, batch_size)
    i += 1
    test_acc.append(test_accuracy)
    del model
    tf.keras.backend.clear_session()

print("----------------------------------------------------")
print("Average test accuracy: ", np.sum(test_acc)/len(test_acc))
print(test_acc)
print("[DONE]...")
