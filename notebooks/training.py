import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def data_generator(files, batchsize):
    while True:
        cnt = 0
        for i in range(len(files)):
            image = np.load(files[i])
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
                yield xtrain, ytrain
        print(xtrain.shape)


def get_predictions(model, files, iteration, save_path, batch_size):
    y_true = []
    y_preds = []
    cnt = 0
    for i in range(len(files)):
        image1 = np.load(files[i])
        label = int(files[i][-5])-1
        if i == 0 or cnt == 0:
            xtest = image1
            ytest = np.array([label])
        else:
            xtest = np.concatenate((xtest, image1), axis=0)
            ytest = np.concatenate((ytest, np.array([label])), axis=0)
        cnt += 1
        if cnt == batch_size or (i == len(files) - 1):
            cnt = 0
            preds = list(np.argmax(model.predict(xtest), axis=1))
            y_preds += preds
            y_true += list(ytest)

    test_accuracy = accuracy_score(y_true, y_preds)
    confusion_mat = confusion_matrix(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='micro')
    f1 = f1_score(y_true, y_preds, average='micro')
    recall = recall_score(y_true, y_preds, average='micro')
    print("-------------------------------------")
    print("Test Accuracy for iteration {}: {}".format(iteration, test_accuracy))
    print("Confussion Matrix: ", confusion_mat)
    print("-------------------------------------")

    f = open(save_path + '/results_' + str(iteration) + ".txt", "a")
    f.write("Test Accuracy score for Iteration - " + str(iteration) + ' : ' + str(test_accuracy))
    f.write('\n')
    f.write("Precision score - " + str(precision))
    f.write('\n')
    f.write("F1 score - " + str(f1))
    f.write('\n')
    f.write("Recall score - " + str(recall))
    f.write('\n')
    f.write("Confussion matrix: \n")
    f.write(str(confusion_mat))
    f.write('\n')
    f.write('--------------------------------------------- \n')
    f.close()
    return test_accuracy
