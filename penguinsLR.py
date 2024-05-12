from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def logisticRegression(features, labels, folds):
    accuracies = []
    classifier = LogisticRegression(multi_class = "multinomial", solver = "saga", max_iter = 100)
    aggregate_confusion_matrix = 0

    for i, (training_ind, testing_ind) in enumerate(folds.split(features)):

        penguinFeatures_split = features[training_ind[0] : training_ind[training_ind.size-1]]    
        penguinLabels_split = labels[training_ind[0] : training_ind[training_ind.size-1]]

        classifier.fit(penguinFeatures_split, np.array(penguinLabels_split).ravel())

        penguinFeatures_test_split = features[testing_ind[0] : testing_ind[testing_ind.size - 1]]
        penguinLabels_test_split = labels[testing_ind[0] : testing_ind[testing_ind.size - 1]]

        penguinLabels_pred = classifier.predict(penguinFeatures_test_split)

        aggregate_confusion_matrix +=  confusion_matrix(penguinLabels_test_split, penguinLabels_pred)

        acc = accuracy_score(penguinLabels_test_split, penguinLabels_pred)
        accuracies.append(acc)

    penguin_Labels = ['Adelie', 'Gentoo', 'Chinstrap']
    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguin_Labels, columns=penguin_Labels)
    return accuracies, average_confusion_matrix 