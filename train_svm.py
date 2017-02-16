import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

def import_and_clean():
    df = pd.read_csv('training_data.csv', header=0)

    df = df[['meanfun' , 'minfun', 'maxfun','label']]
    df['label'] = df['label'].map({'female': 0, 'male': 1}).astype(int)
    return df

def train_svm(input_df):
    x = input_df[['meanfun' , 'minfun', 'maxfun']].values
    y = input_df['label'].values
    svc = SVC(kernel='rbf')
    training, testing, training_result, testing_result = train_test_split(x, y, test_size=0.1, random_state=1)
    scores = cross_val_score(svc, training, training_result, cv=10, scoring='accuracy')
    print scores
if __name__ == '__main__':
    df =import_and_clean()
    train_svm(df)