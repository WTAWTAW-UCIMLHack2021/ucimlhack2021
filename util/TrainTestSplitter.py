import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from util import transform

class TrainTestSplitter:
    def __init__(self, dataset: pd.DataFrame):
        self._data = dataset
        # Keys are misconception IDs, values are the misconception in English
        # self._misconception_map = pd.Series(dataset.misconception.values, index = dataset.misconception_id).to_dict()
            
    def transform_split(self, test_size=0.25):
        """
        Splits dataset into random train and test subsets such that each row represents a unique tweet. 
        The dataset is split so that the proportion of misconception-spreading tweets is the same across 
        training and testing sets.
        
        Parameters:
            test_size(float): The proportion of the dataset to include in the test split
            
        Returns:
            (X_train, X_test, y_train, y_test)
            X_train(Series): Tweet text (str)
            X_test(Series): Tweet text (str)
            y_train(Series): Whether the tweet spreads a misconception (1/0)
            y_test(Series): Whether the tweet spreads a misconception (1/0)
        """
        # Transform dataframe so each row is a unique tweet
        df = transform(self._data)
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # Get indices of items to include in the train/testing sets
        train_index, test_index = tuple(sss.split(df.tweet, df.misconception))[0]
            
        # Get the correct items to include in the train/testing sets
        X_train = df.tweet.take(train_index)
        X_test = df.tweet.take(test_index)
        y_train = df.misconception.take(train_index)
        y_test = df.misconception.take(test_index)
        
        return X_train, X_test, y_train, y_test
        
# Only executed when the file is run directly
if __name__ == '__main__':
    dataset = pd.read_csv('./data/covid_lies.csv')
    
    splitter = TrainTestSplitter(dataset)
    
    X_train, X_test, y_train, y_test = splitter.transform_split()
    
    print(f"Proportion of Misconceptions in Train: {sum(y_train)/len(y_train)} | Proportion of Misconceptions in Test: {sum(y_test)/len(y_test)}")
    print(f"Number of Tweets in Train: {len(y_train)} | Number of Tweets in Test: {len(y_test)}")

    # TODO: Evaluate the number of unseen misonceptions in train + number of unseen misconceptions in test
    