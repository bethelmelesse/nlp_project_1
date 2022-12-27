from nltk.corpus import twitter_samples                    # our dataset
import numpy as np


def accessing_data():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    all_tweets = all_positive_tweets + all_negative_tweets

    all_positive_labels = np.ones((len(all_positive_tweets)))
    all_negative_labels = np.zeros((len(all_negative_tweets)))
    all_labels = np.append(all_positive_labels, all_negative_labels)

    classes = ["negative", "positive"]  

    return all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels,  classes


# split the data into two piece, one for training and one for testing (validation set)
def train_dataset(all_positive_tweets, all_negative_tweets):

    train_positives_x = all_positive_tweets[:4000]
    train_negatives_x = all_negative_tweets[:4000]
    train_x = train_positives_x + train_negatives_x

    train_positives_y = np.ones((len(train_positives_x)))
    train_negatives_y = np.zeros((len(train_negatives_x)))
    train_y = np.append(train_positives_y,  train_negatives_y)

    return train_x, train_y


def test_dataset(all_positive_tweets, all_negative_tweets):

    test_positives_x = all_positive_tweets[4000:]
    test_negatives_x = all_negative_tweets[4000:]
    test_x = test_positives_x + test_negatives_x

    test_positives_y = np.ones((len(test_positives_x)))
    test_negatives_y = np.zeros((len(test_negatives_x)))
    test_y = np.append(test_positives_y, test_negatives_y)

    return test_x, test_y

all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels, classes = accessing_data()
train_x, train_y = train_dataset(all_positive_tweets, all_negative_tweets)
test_x, test_y = test_dataset(all_positive_tweets, all_negative_tweets)

# print(train_x[0]) 
# print(train_x[4001]) 

from preprocessing import *
print(process_tweet(train_x[0]))
print(process_tweet(train_x[4001])) 