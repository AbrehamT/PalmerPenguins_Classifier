import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def neuralNetwork(features, labels, folds):
    accuracies = []
    currFold = 1
    aggregate_confusion_matrix = 0

    for train, test in folds.split(features):
        featuresTrain= features.iloc[train, :]
        labelsTrain = labels.iloc[train]

        featuresTest = features.iloc[test, :]
        labelsTest = labels.iloc[test]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(3)])
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
              optimizer=opt,
              loss=loss_fn,
              metrics=["accuracy"])
        
        print(f"Fold {currFold}:")
        trainNN = model.fit(featuresTrain, labelsTrain, batch_size=2, epochs=10)
        print("\nTesting Accuracy:")
        history = model.evaluate(featuresTest, labelsTest, batch_size=2, verbose=2)
        accuracies.append(history[1])

        # Making Predictions for Confusion Matrix
        labelsPredicted = np.argmax(model.predict(featuresTest), axis = 1)
        aggregate_confusion_matrix +=  confusion_matrix(labelsTest, labelsPredicted)
        print("\n")
        currFold += 1

    
    penguin_Labels = ['Adelie', 'Gentoo', 'Chinstrap']
    average_confusion_matrix = pd.DataFrame((aggregate_confusion_matrix / folds.n_splits),  index=penguin_Labels, columns=penguin_Labels)

    return accuracies, average_confusion_matrix
