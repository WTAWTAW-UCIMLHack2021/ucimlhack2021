import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import random
import math
# from collections import namedtuple

from util import transform

covidlies = pd.read_csv('./data/covid_lies.csv')

def stratify_split(tweet_ids: list, is_misconception: list, test_size=0.25):
    """
    The dataset is split so that the proportion of misconception-spreading tweets is the same across 
    training and testing sets. Does not seem to give different results for each test_size.

    Parameters:
        tweet_ids(pd.Series or list<int>-like): List t where t[i] represents a unique tweet id
        is_misconception(pd.Series or list<bool>-like):  List m where m[i] represents whether the tweet t[i] spreads a misconception
        test_size(float): The proportion of the dataset to include in the test split

    Returns:
        prop_diff(float): The difference in proportions (test - train) of tweets marked as misconception (~0.00)
        train_indices(list<int>): The indices corresponding to tweets that should be included in the train set
        test_indices(list<int>): The indices corresponding to tweets that should be included in the test set
    """
    tweet_ids = list(tweet_ids)
    is_misconception = list(is_misconception)
    
    assert(len(tweet_ids) == len(is_misconception)), "Lists storing tweets IDs and misconception status must be the same length"

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    # Get indices of items to include in the train/testing sets
    train_indices, test_indices = tuple(sss.split(tweet_ids, is_misconception))[0]
        
    prop_misconception_train = sum([is_misconception[i] for i in train_indices]) / len([is_misconception[i] for i in train_indices])
    prop_misconception_test= sum([is_misconception[i] for i in test_indices]) / len([is_misconception[i] for i in test_indices])
    prop_diff = prop_misconception_test - prop_misconception_train

    return prop_diff, train_indices, test_indices
    
def hide_split(tweet_ids: list, is_misconception: list, test_size=0.25, prop_to_hide=0.2):
    """
    The dataset is split so that tweets spreading a given proportion of misconceptions are excluded from the train set.
    First, a random proportion of misconceptions that have at least one tweet taking a stance towards it are selected to be hidden.
    Any tweet that takes a stance towards these selected misconceptions only appears in the test set. 
    The end result is that a given proportion of misconceptions (with relevant tweets) are never seen during training. 
        
    Parameters:
        tweet_ids(pd.Series or list<int>-like): List t where t[i] represents a unique tweet id
        is_misconception(pd.Series or list<bool>-like):  List m where m[i] represents whether the tweet t[i] spreads a misconception
        test_size(float): The proportion of the dataset to include in the test split
        prop_to_hide(float): The proportions of tweets taking a stance to exclude from the train set

    Returns:
        prop_diff(float): The difference in proportions (test - train) of tweets marked as misconception (range from ~0.01-0.2)
        train_indices(list<int>): The indices corresponding to tweets that should be included in the train set
        test_indices(list<int>): The indices corresponding to tweets that should be included in the test set
    """
    assert(len(tweet_ids) == len(is_misconception)), "Lists storing tweets IDs and misconception status must be the same length"

    tweet_ids = list(tweet_ids)
    is_misconception = list(is_misconception)

    # Set of all misconceptions where a tweet has taken a stance towards it
    stanced_misconceptions = list(covidlies[covidlies.label != 'na'].misconception_id.unique())
    
    # Select a percentage of misconceptions to hide during training
    num_to_sample = math.ceil(len(stanced_misconceptions) * prop_to_hide)
    misconceptions_to_hide = random.sample(stanced_misconceptions, k = num_to_sample)

    # Tweets that take a stance towards one of the misconceptions we want to hide during training
    tweets_to_exclude = set(covidlies[(covidlies.misconception_id.isin(misconceptions_to_hide) & (covidlies.label != 'na'))].tweet_id)
    # Tracks whether each tweet takes a stance towards a misconception we want to hide
    groups = [1 if tweet_id in tweets_to_exclude else 0 for tweet_id in tweet_ids]

    # Put all three columns into one dataframe
    df = pd.DataFrame(data={'tweet_ids': tweet_ids, 'is_miconception': is_misconception, 'groups': groups})
    # Shuffle DataFrame and sort tweets we want to hide to the top
    sorted = df.sample(frac=1).sort_values(by='groups', ascending=False)
    
    # The dividing row between the test set (above) and the train set (below)
    split_index = math.ceil(test_size * len(sorted))
    assert(split_index > sum(groups)), "Test size is too small! Hidden misconceptions will leak into train set"
    
    train_indices = list(sorted.iloc[split_index:].index)
    test_indices = list(sorted.iloc[:split_index].index)

    # Calculate the difference in proportions (test - train) of tweets marked as misconception
    prop_misconception_train = sum([is_misconception[i] for i in train_indices]) / len([is_misconception[i] for i in train_indices])
    prop_misconception_test= sum([is_misconception[i] for i in test_indices]) / len([is_misconception[i] for i in test_indices])
    prop_diff = prop_misconception_test - prop_misconception_train

    assert(len(train_indices) + len(test_indices) == len(tweet_ids)), "Train/Test split was not executed correctly"
    
    return prop_diff, train_indices, test_indices
    

#     def _lookup(self, t_id):
#         """
#         Looks up the misconceptions a tweet took a stance towards and whether the tweet conveyed ANY misconception
        
#         Returns:
#             TweetData: namedtuple('id', 'misconception_ids', 'label)
#                 id(int): The unique id identifying a tweet
#                 misconception_ids(set): Misconception ids the tweet took a stance towards
#                 label(bool): If the tweet has taken a positive stance towards any misconception
#         """
#         TweetData = namedtuple('TweetData', ['id', 'misconception_ids', 'label'])
        
#         tweet = covidlies[(covidlies.tweet_id == t_id) & (covidlies.label != 'nan')][['tweet_id', 'misconception_id', 'label']]
#         t_data = TweetData(t_id, set(tweet.misconception_id), 'pos' in tweet.label.unique())
        
#         return t_data
    
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

# EXAMPLE USAGE
# Only executed when the file is run directly
if __name__ == '__main__':
    transformed = transform(covidlies)
    s_prop_diff, s_train_indices, s_test_indices = stratify_split(transformed.tweet_id, transformed.misconception)
    h_prop_diff, h_train_indices, h_test_indices = hide_split(transformed.tweet_id, transformed.misconception)
    
    print(f'Stratified Split Prop Diff: {s_prop_diff} Hide Split Prop Diff: {h_prop_diff}')

    # Create training and test sets using hide split indices
    X_train = transformed.tweet.take(h_train_indices)
    X_test = transformed.tweet.take(h_test_indices)
    y_train = transformed.misconception.take(h_train_indices)
    y_test = transformed.misconception.take(h_test_indices)
    
    # print(f"Proportion of Misconceptions in Train: {sum(y_train)/len(y_train)} | Proportion of Misconceptions in Test: {sum(y_test)/len(y_test)}")
    print(f"Number of Tweets in Train: {len(y_train)} | Number of Tweets in Test: {len(y_test)}")
    