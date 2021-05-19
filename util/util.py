import pandas as pd

def transform(data) -> pd.DataFrame:
    """
    Returns a dataframe of unique tweets and whether it spreads a Covid-19 misconception
    
    Parameters:
        data(DataFrame): A DataFrame with CovidLies data
    
    Columns:
        tweet_id(int): The unique ID identifying a tweet
        tweet(str): The text of a unique tweet
        misconception(bool): Whether the text spreads a misonception 
    """
    # Set of tweet IDs that spread a misconception
    misconception_tweets = set(data[data.label == 'pos'].tweet_id.unique())
    # Two columns of unique tweets        
    df = data[['tweet_id', 'tweet']].drop_duplicates('tweet_id')
    # Indicate whether each tweet spread a misconception
    df['misconception'] = df.tweet_id.apply(lambda id: id in misconception_tweets)
    
    return df
