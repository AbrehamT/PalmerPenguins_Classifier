from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

def SVM(features, labels, folds):
    accuracies = []
    currFold = 1
    aggregate_confusion_matrix = 0

    for train, test in folds.split(features):
        featuresTrain= features.iloc[train, :]
        labelsTrain = labels.iloc[train]

        featuresTest = features.iloc[test, :]
        labelsTest = labels.iloc[test]
        
        model = SVC(kernel='linear')
        model.fit(featuresTrain, np.array(labelsTrain).ravel())

        labelsPredict = model.predict(featuresTest)
        aggregate_confusion_matrix +=  confusion_matrix(labelsTest, labelsPredict)

        accuracy = accuracy_score(labelsTest, labelsPredict) * 100
        accuracies.append(accuracy)

    penguin_Labels = ['Adelie', 'Gentoo', 'Chinstrap']
    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguin_Labels, columns=penguin_Labels)
    
    return accuracies, average_confusion_matrix 