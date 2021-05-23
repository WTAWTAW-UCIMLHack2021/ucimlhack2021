import pandas as pd

def transform(data, unique_by_tweet_contents=True) -> pd.DataFrame:
    """
    Returns a dataframe of unique tweets and whether it spreads a Covid-19 misconception
    
    IMPORTANT: Some tweets have a different ID but have the same contents:
    ex.) Tweet ID A: "Hello There!"
         Tweet ID B: "Hello There!"
         Tweet ID C: "Hello There :)"
         
         If we consider tweets unique by their contents (the default): A and C would be in the transformed dataframe
         If we consider tweets unique by their ID: A, B, and C would be in the transformed dataframe.
         
    This approach is limited in that we cannot account for tweets that are similar but not identical (see A and C in example). However,
    we estimate such an approach eliminates upwards of 97% of semantically identical tweets from by counting the number of total tweets
    that are highly related (0.95 > Jaccard Set similarity -- see util/minhash.ipynb)
    
    Parameters:
        data(DataFrame): A DataFrame with CovidLies data
        unique_by_tweet_contents(bool): Whether to consider a tweet unique by its ID or by its contents 
    
    Columns:
        tweet_id(int): The unique ID identifying a tweet
        tweet(str): The text of a unique tweet
        misconception(bool): Whether the text spreads a misonception 
    """
    if unique_by_tweet_contents:
        # Set of tweet contents that spread a misconception
        misconception_tweets = set(data[data.label == 'pos'].tweet.unique())
        # Two columns of unique tweet contents       
        df = data[['tweet_id', 'tweet']].drop_duplicates('tweet')
        # Indicate whether each tweet content spread a misconception
        df['misconception'] = df.tweet.apply(lambda id: id in misconception_tweets)
    else:
        # Set of tweet IDs that spread a misconception
        misconception_tweets = set(data[data.label == 'pos'].tweet_id.unique())
        # Two columns of unique tweet IDs       
        df = data[['tweet_id', 'tweet']].drop_duplicates('tweet_id')
        # Indicate whether each tweet ID spread a misconception
        df['misconception'] = df.tweet_id.apply(lambda id: id in misconception_tweets)

    return df
