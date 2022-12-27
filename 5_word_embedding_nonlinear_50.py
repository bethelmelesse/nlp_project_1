'''WORD EMBEDDING MLP'''

import torch
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import gensim.downloader as api

from preprocessing import *
from preparing_data import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

word_vec_model = api.load("glove-wiki-gigaword-50")

'''PREPROCESS STAGE'''

# prepare the datasets 
all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels, classes = accessing_data()
train_x, train_y = train_dataset(all_positive_tweets, all_negative_tweets)
test_x, test_y = test_dataset(all_positive_tweets, all_negative_tweets)

# process training data
processed_train_x = process_data(train_x, stem=False)
processed_test_x = process_data(test_x, stem=False)

def extract_features_word_embedding(tweet):
    tweet_vector = []
    for word in tweet:
        if word in word_vec_model:
            word_vector = word_vec_model[word]
            tweet_vector.append(word_vector)

    if len(tweet_vector) == 0:
        tweet_vector.append(np.zeros(50))
        

    tweet_vector = np.array(tweet_vector)
    tweet_vector = np.mean(tweet_vector, axis=0)
    return tweet_vector

def convert_to_tensors_word_embedding(processed_x, y, device):
    # extracting features 
    x = []
    for tweet in processed_x:
        single_tweet = extract_features_word_embedding(tweet)
        x.append(single_tweet)

    x = torch.tensor(np.array(x), dtype=torch.float, device=device)
    y = torch.tensor(np.array(y), dtype=torch.float, device=device)

    return x, y

# convert to tensor
x_for_train, train_y = convert_to_tensors_word_embedding(processed_train_x, train_y, device)
x_for_test, test_y = convert_to_tensors_word_embedding(processed_test_x, test_y, device)


''' OUR MODEL '''
class LinearRegressionModel(torch.nn.Module):         # Initializing our model.
 
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(50, 50, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1, bias=True),
        )
        
 
    def forward(self, x):                           # Declaring the forward pass.
        y_pred = self.linear_relu_stack(x)
        return y_pred

our_model = LinearRegressionModel().to(device=device)               # object of this model     
the_loss = torch.nn.MSELoss()                                       # the loss criteria
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)     # optimizer


''' ACCURACY METRIC'''
def accuracy(pred_y, ground_truth_y):
    pred_y = pred_y.cpu().detach().numpy()
    for i in range(len(pred_y)):
        if pred_y[i] >= 0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = 0
    
    correct = pred_y - ground_truth_y.cpu().detach().numpy()
    correct_freq = np.count_nonzero(correct==0)
    total_freq = len(correct)

    accuracy_value = (correct_freq / total_freq) * 100
    return accuracy_value


''' TRAINING STAGE '''
def train():
    '''Training stage'''
    # Perform a forward pass bypassing our data and finding out the predicted value of y
    # Compute the loss using MSE.
    # Reset all the gradients to 0, perform a backpropagation and then, update the weights.
    EPOCH = 250
    for epoch in range(EPOCH):
        optimizer.zero_grad()                                 # Zero gradients
        pred_y = our_model(x_for_train)                       # Forward pass: Compute predicted y by passing x to the model
        pred_y = torch.squeeze(pred_y, dim=1)                 # (8000, 50) ------> (8000, 1)
        # pred_y = torch.squeeze(pred_y, dim=1)                 # (8000, 1)    ------> (8000)
        loss = the_loss(pred_y, train_y)                      # Compute and print loss
        loss.backward()                                       # perform a backward pass
        optimizer.step()                                      # update the weights.

        if (epoch %50 == 0):
            accuracy_value = accuracy(pred_y, train_y)            # not really needed
            print('\033[35m' + f'epoch {epoch}, train loss {loss.item()}, train accuracy {accuracy_value}' + '\33[97m')
            test()


''' TESTING STAGE '''
def test():
    our_model.eval()
    with torch.no_grad():
        pred_y = our_model(x_for_test)                         # make predicitoin 
        pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 50) ------> (2000, 1)
        # pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 1)    ------> (2000)
        loss = the_loss(pred_y, test_y)                        # calculate test loss - not really needed

        accuracy_value = accuracy(pred_y, test_y)              # to check model performance 
        print()
        print('\033[94m' + f'test loss {loss}, test accuracy {accuracy_value}' + '\33[97m')
    our_model.train()


train()
test()