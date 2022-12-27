import numpy as np
from preprocessing import process_tweet


def build_freqs(tweets, ys):
    ''' Build frequencies
    Input:
        tweets: a list of tweets
        ys: an m * 1 array with the sentiment label of each tweet (either 0 or 1)
    Output:
        freqs: a dict mapping each pair (word, sentiment) to its frequency
        '''
    # convert np array to list since zip needs an iterable
    # the squeeze is necessary or the list ends with one element
    # also not that this is just a NOP if ys is already a list
    yslist = np.squeeze(ys).tolist()

    # start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs

# freqs = build_freqs(tweets, ys)           # {(word, sentiment): freq}