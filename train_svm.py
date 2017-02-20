import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def import_and_clean():
    df = pd.read_csv('training_data.csv', header=0)

    df = df[['meanfun' , 'minfun', 'maxfun','label']]
    df['label'] = df['label'].map({'female': 0, 'male': 1}).astype(int)
    return df

def parameter_tuning_svm(input_df):
    x = input_df[['meanfun' , 'minfun', 'maxfun']].values
    y = input_df['label'].values
   #svc = SVC(kernel='linear')

    #segmenting data set and cross validation
    training, testing, training_result, testing_result = train_test_split(x, y, test_size=0.4, random_state=1)
    # scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
    # print scores.mean()

    #Tuning C value
    c_vals = list(range(1,30))
    accuracy_vals = []
    for val in c_vals:
        svc = SVC(kernel='linear', C=val)
        scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
        accuracy_vals.append(scores.mean())

    plt.plot(c_vals, accuracy_vals)
    plt.xticks(np.arange(0,30,2))
    plt.xlabel('C values')
    plt.ylabel('Mean Accuracies')
    plt.show()

    optimal_cval = c_vals[accuracy_vals.index(max(accuracy_vals))]
    print optimal_cval

if __name__ == '__main__':
    df =import_and_clean()
    parameter_tuning_svm(df)