import numpy as np
import re                                       # library for regular expression operations
import string                                   # for string operations

from nltk.corpus import stopwords                # module for stop words that come with NLTK
from nltk.stem import PorterStemmer              # module for stemming
from nltk.tokenize import TweetTokenizer         # module for tokenizing strings
import torch

# nltk.download("stopwords")                    # download the stopwords from NLTK

def process_tweet(tweet, stem=True):

    tweet = re.sub(r'^RT[\s]+', '', tweet)                    # Remove old style retweet text "RT"
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)         # remove hyperlinks
    tweet = re.sub(r'#', '', tweet)                           # remove hashtags # only removing the hash # sign from the word
    # :-(  :(  :d :-) :)
    tweet = re.sub(r':-\(', '', tweet)
    tweet = re.sub(r':\(', '', tweet)
    tweet = re.sub(r':d', '', tweet)
    tweet = re.sub(r':-\)', '', tweet)
    tweet = re.sub(r':\)', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)  # instantiate tokenizer class
    tweet_tokens = tokenizer.tokenize(tweet)  # Tokenize tweet

    stopwords_english = stopwords.words('english')  # Import the english stop words list from NLTK

    tweets_clean = []
    for word in tweet_tokens:                             # Go through every word in your tokens list
        if (word not in stopwords_english and             # remove stopwords
                word not in string.punctuation):          # remove punctuation
            tweets_clean.append(word)

    if stem==True:
        stemmer = PorterStemmer()                             # Instantiate stemming class)
        tweets_stem = []                                      # create an empty list to store the stems
        for word in tweets_clean:
            stem_word = stemmer.stem(word)                    # stemming word
            tweets_stem.append(stem_word)                     # append to the list
        processed_tweet = tweets_stem
    else:
        processed_tweet = tweets_clean

    return processed_tweet


def process_data(data, stem=True):
    processed_data = []
    for tweet in data:
        processed_tweet = process_tweet(tweet, stem=stem)
        processed_data.append(processed_tweet)
    return processed_data
    

def extract_features(tweet, freqs):                              
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    
    x = np.zeros((1, 2))                     # 3 elements in the form of a 1 x 3 vector   
        
    # loop through each word in the list of words
    for word in tweet:

        # increment the word count for the positive label 1
        if (word,1) in freqs:
            x[0,0] += 1
        
        # increment the word count for the negative label 0
        if (word,0) in freqs:
            x[0,1] += 1
        
    assert(x.shape == (1, 2))
    return x


def convert_to_tensors(processed_x, y, freqs, device):
    # extracting features 
    x = []
    for tweet in processed_x:
        single_tweet = extract_features(tweet, freqs)
        x.append(single_tweet)

    x = torch.tensor(np.array(x), dtype=torch.float, device=device)
    y = torch.tensor(np.array(y), dtype=torch.float, device=device)

    return x, y

# process_data(data, stem=True/False)           # used in linear regression, word embedding linear, word embedding nonlinear
# extract_features(tweet, freqs)                # used in linear regression
# convert_to_tensor                             # used in linear regression