import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class TrainTestSplitter:
    def __init__(self, dataset: pd.DataFrame):
        self._data = dataset
        # Keys are misconception IDs, values are the misconception in English
        # self._misconception_map = pd.Series(dataset.misconception.values, index = dataset.misconception_id).to_dict()
            
    def transform(self) -> pd.DataFrame:
        """
        Returns a dataframe of unique tweets and whether it spreads a Covid-19 misconception
        
        Columns:
            tweet_id(int): The unique ID identifying a tweet
            tweet(str): The text of a unique tweet
            misconception(bool): Whether the text spreads a misonception 
        """
        # Set of tweet IDs that spread a misconception
        misconception_tweets = set(self._data[self._data.label == 'pos'].tweet_id.unique())
        # Two columns of unique tweets        
        df = self._data[['tweet_id', 'tweet']].drop_duplicates('tweet_id')
        # Indicate whether each tweet spread a misconception
        df['misconception'] = df.tweet_id.apply(lambda id: id in misconception_tweets)
        
        return df
    
    def transform_split(self, test_size=0.25):
        """
        Splits dataset into random train and test subsets such that each row represents a unique key. 
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
        df = self.transform()
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # Get indices of items to include in the train/testing sets
        train_index, test_index = tuple(sss.split(df.tweet, df.misconception))[0]
            
        # Get the correct items to include in the train/testing sets
        X_train = df.tweet.take(train_index)
        X_test = df.tweet.take(test_index)
        y_train = df.misconception.take(train_index)
        y_test = df.misconception.take(test_index)
        
        return X_train, X_test, y_train, y_test
        
# Only run when the file is run directly
if __name__ == '__main__':
    dataset = pd.read_csv('./data/covid_lies.csv')
    
    splitter = TrainTestSplitter(dataset)
    
    X_train, X_test, y_train, y_test = splitter.transform_split()
    
    print(f"Proportion of Misconceptions in Train: {sum(y_train)/len(y_train)} | Proportion of Misconceptions in Test: {sum(y_test)/len(y_test)}")
    print(f"Number of Tweets in Train: {len(y_train)} | Number of Tweets in Test: {len(y_test)}")

    # TODO: Evaluate the number of unseen misonceptions in train + number of unseen misconceptions in test
    