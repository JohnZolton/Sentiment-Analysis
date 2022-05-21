#twitter sentiment analyzer
from ast import Lambda
from imaplib import _Authenticator
import tweepy
import textblob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import keys

api_key = keys.api_key
api_key_secret = keys.api_key_secret
access_token = keys.access_token
access_token_secret = keys.access_token_secret


authenticator = tweepy.OAuthHandler(api_key, api_key_secret)
authenticator.set_access_token(access_token, access_token_secret)

api = tweepy.API(authenticator, wait_on_rate_limit=True)

topic = 'bitcoin'

search = f'#{topic} -filter:retweets'

tweet_cursor = tweepy.Cursor(api.search_tweets, q=search, lang='en', tweet_mode ='extended').items(100) #add since and until

tweets = [tweet.full_text for tweet in tweet_cursor]

tweets_df = pd.DataFrame(tweets, columns=['Tweets'])
print('original ', tweets_df)

for _,row in tweets_df.iterrows():
    row['Tweets'] = re.sub('http\S+', '', row['Tweets'])
    row['Tweets'] = re.sub('#\S+', '', row['Tweets'])
    row['Tweets'] = re.sub('@\S+', '', row['Tweets'])
    row['Tweets'] = re.sub('\\n+', '', row['Tweets'])

print('cleaned up ',tweets_df)

tweets_df['Polarity'] = tweets_df['Tweets'].map(lambda tweet: textblob.TextBlob(tweet).sentiment.polarity)
tweets_df['Result'] = tweets_df['Polarity'].map(lambda pol: '+' if pol > 0 else '-')

print('updated\n', tweets_df)

positive = tweets_df[tweets_df.Result == '+'].count()['Tweets']
negative = tweets_df[tweets_df.Result == '-'].count()['Tweets']

print('postive\n', positive)
print('negative\n', negative)

plt.bar([0,1], [positive, negative], label = ['Positive', 'Negative'], color=['green', 'red'])

plt.show()