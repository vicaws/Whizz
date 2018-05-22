import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def compute_predictionScore(expectations, map_predictions, n_components, 
                            criterion='deviation'):
    '''Evaluate the score of prediction by specified method.

    Parameters
    ----------
    expectations: array-like, shape=(n_sample, 1)
        The expected/actual classification labels (1 for churn and 0 otherwise) 
        for samples.

    map_predictions: dictionary, 
            key=int, predictions trial number;
            value=array, predictions, shape=(n_sample, 1)
        The collection of different trials of predicted classification labels
        for samples.

    n_components: int
        Number of components/clusterings identified.

    criterion: string
        Definition of score calculation. 

    Returns
    -------
    map_score: dictionary, 
            key=int, predictions trial number;
            value=float, score of the predictions
        The collection of scores of predictions.
    '''

    map_score = {}
    base = expectations.sum()/expectations.shape[0]
    for trial, predictions in map_predictions.items():
        num_pupil = []
        num_churn = []

        for i in range(0, n_components):
            idx_pupils = np.where(predictions==i)[0]
            if idx_pupils.size > 0: # remove empty groups
                num_pupil.append(idx_pupils.shape[0])
                num_churn.append(expectations[idx_pupils].sum())
        
        df = pd.DataFrame({'num_pupil':num_pupil, 'num_churn':num_churn})
        df = df.assign(rate_churn=df.num_churn/df.num_pupil)

        if criterion == 'deviation':
            score = (abs(df.rate_churn-base) * df.num_pupil).sum()
        elif criterion == 'variance':
            score = (((df.rate_churn-base) * df.num_pupil)**2).sum()
        elif criterion == 'max_rate':
            score = max(df.rate_churn-base)
        elif criterion == 'max_number':
            score = max((df.rate_churn-base)*df.num_pupil)
        elif criterion == 'min_rate':
            score = -min(df.rate_churn-base)
        elif criterion == 'min_number':
            score = -min((df.rate_churn-base)*df.num_pupil)
        else:
            raise IndexError(f'The input criterion \'{criterion}\' has not ',
                             'been defined yet!')
        
        map_score[trial] = score
    
    return map_score

def confusion_matrix(expectations, predictions, classes,
                     normalise=True,
                     size=5,
                     title='Confusion Matrix',
                     cmap=plt.cm.Blues):
    '''Return and plot the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'
    '''
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(expectations, predictions)
    np.set_printoptions(precision=2)

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalised confusion matrix")
    else:
        print("Confusion matrix, without normalisation")

    fig = plt.figure(figsize=(size,size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()

    return cm
