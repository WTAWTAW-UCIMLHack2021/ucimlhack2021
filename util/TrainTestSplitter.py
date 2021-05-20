import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import namedtuple

from util import transform

covidlies = pd.read_csv('./data/covid_lies.csv')

class TrainTestSplitter:
    def __init__(self):
        pass
        # Keys are misconception IDs, values are the misconception in English
        # self._misconception_map = pd.Series(dataset.misconception.values, index = dataset.misconception_id).to_dict()
            
    def split(self, tweet_ids: list, is_misconception: list, test_size=0.25):
        """
        Splits dataset into random train and test subsets such that each row represents a unique tweet. 
        The dataset is split so that the proportion of misconception-spreading tweets is the same across 
        training and testing sets.
        
        Parameters:
            test_size(float): The proportion of the dataset to include in the test split
            
        Returns:
            train_indices(np.ndarray): The indices corresponding to tweets that should be included in the train set
            test_indices(np.ndarray): The indices corresponding to tweets that should be included in the test set
        """
        assert(len(tweet_ids) == len(is_misconception)), "Lists storing tweets IDs and misconception status must be the same length"
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # Get indices of items to include in the train/testing sets
        train_indices, test_indices = tuple(sss.split(tweet_ids, is_misconception))[0]
            
        return train_indices, test_indices
    
    def _lookup(self, t_id):
        """
        Looks up the misconceptions a tweet took a stance towards and whether the tweet conveyed ANY misconception
        
        Returns:
            TweetData: namedtuple('id', 'misconception_ids', 'label)
                id(int): The unique id identifying a tweet
                misconception_ids(set): Misconception ids the tweet took a stance towards
                label(bool): If the tweet has taken a positive stance towards any misconception
        """
        TweetData = namedtuple('TweetData', ['id', 'misconception_ids', 'label'])
        
        tweet = covidlies[(covidlies.tweet_id == t_id) & (covidlies.label != 'nan')][['tweet_id', 'misconception_id', 'label']]
        t_data = TweetData(t_id, set(tweet.misconception_id), 'pos' in tweet.label.unique())
        
        return t_data
    
    # def _hide(self, tweet_ids, train_indices, test_indices):
        
        # def move(old_train_index:int, old_label:bool,  new_train_indices, new_test_indices) -> None:
            
        #     for test_index in new_test_indices: 
        #         tweet_id = tweet_ids[test_index]
        #         tweet_data = self._lookup(tweet_id)
            
        #         # print(tweet_id)
            
        #         if old_label != tweet_data.label:
        #             continue

        #         # Adresses misconception that should be hidden
        #         if tweet_data.misconception_ids & misconceptions_to_hide != set():
        #             continue
                
        #         new_train_indices.append(test_index)
        #         return
                
        #     new_test_indices.append(old_train_index)            
        
        # new_train_indices = []
        # new_test_indices = list(test_indices.copy())
        
        # misconceptions_to_hide = set(covidlies[covidlies.label != 'na'].misconception_id.unique())
        
        # for train_index in train_indices:
        #     tweet_id = tweet_ids[train_index]
        #     tweet_data = self._lookup(tweet_id)
            
        #     print(tweet_id)
            
        #     if tweet_data.label == False:
        #         new_train_indices.append(train_index)
        #         continue
            
        #     if tweet_data.misconception_ids & misconceptions_to_hide == set():
        #         new_train_indices.append(train_index)
        #         continue
            
        #     print(tweet_id)

            
        #     move(train_index, tweet_data.label, new_train_indices, new_test_indices)
        
        # return new_train_indices, new_test_indices
            
# Only executed when the file is run directly
if __name__ == '__main__':
    splitter = TrainTestSplitter()
    
    transformed = transform(covidlies)
    train_indices, test_indices = splitter.split(transformed.tweet_id, transformed.misconception)
    
    X_train = transformed.tweet.take(train_indices)
    X_test = transformed.tweet.take(test_indices)
    y_train = transformed.misconception.take(train_indices)
    y_test = transformed.misconception.take(test_indices)
    
    print(f"Proportion of Misconceptions in Train: {sum(y_train)/len(y_train)} | Proportion of Misconceptions in Test: {sum(y_test)/len(y_test)}")
    print(f"Number of Tweets in Train: {len(y_train)} | Number of Tweets in Test: {len(y_test)}")

    # TODO: Evaluate the number of unseen misonceptions in train + number of unseen misconceptions in test
    