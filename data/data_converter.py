import pandas as pd
import csv

covidlies = pd.read_csv('covid_lies.csv')

if __name__ == '__main__':
    tweets = []
    for i in covidlies.keys():
        print(i)
    for i, tweet in enumerate(covidlies['tweet']):
        tweetlist = [word for word in tweet.split(" ") if word != "@USERNAME"
                     and "https://" not in word]
        newTweet = ' '.join(tweetlist)
        tweets.append([str(covidlies[k][i]) if k != 'tweet' else newTweet for k in covidlies.keys()])


with open('covid_lies_data.csv', 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(list(covidlies.keys()))
    csvwriter.writerows(tweets)