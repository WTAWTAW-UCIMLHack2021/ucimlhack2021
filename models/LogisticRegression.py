import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from util.util import transform
from BagOfWords import BagOfWords, data_split

if __name__ == "__main__":
    learner = LogisticRegression()
    pure_df = pd.read_csv('../data/covid_lies_processed.csv')
    pure_df = transform(pure_df)

    X_train, Y_train, X_test, Y_test = data_split(pure_df, 'tweet', 'tweet', 'misconception')

    train = BagOfWords(X_train, Y_train)
    training_words = train._words
    X_train_vector = train.x_data()

    test = BagOfWords(X_test, Y_train, words=training_words)
    X_test_vector = test.x_data()

    learner.fit(X_train_vector, Y_train)
    accuracy = learner.score(X_test_vector, Y_test)
    print(accuracy)

    y_hat = learner.predict(X_test_vector)
    print(sum(y_hat))
    print(sum(Y_test))
    print(precision_recall_fscore_support(Y_test, y_hat))