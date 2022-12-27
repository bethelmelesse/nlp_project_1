'''WORD EMBEDDING MLP with regularisaton and cross entropy loss'''

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

word_vec_model = api.load("glove-twitter-200")

'''PREPROCESS STAGE'''
def shuffle_list(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1, res2

# prepare the datasets 
all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels, classes = accessing_data()
train_x, train_y = train_dataset(all_positive_tweets, all_negative_tweets)
train_x, train_y = shuffle_list(train_x, train_y)
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
        tweet_vector.append(np.zeros(200))
        

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
    y = torch.tensor(np.array(y), dtype=torch.long, device=device)

    return x, y

# convert to tensor
x_for_train, train_y = convert_to_tensors_word_embedding(processed_train_x, train_y, device)
x_for_test, test_y = convert_to_tensors_word_embedding(processed_test_x, test_y, device)


''' OUR MODEL '''
# Initializing our model.
# Declaring the forward pass.
class LinearRegressionModel(torch.nn.Module):
 
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(200, 50, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(50, 50, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(50, 2, bias=True),
        )
        
 
    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred

our_model = LinearRegressionModel().to(device=device)               # object of this model     
the_loss = torch.nn.CrossEntropyLoss()                                       # the loss criteria
optimizer = torch.optim.AdamW(our_model.parameters(), lr = 0.001, weight_decay=0.01)     # select the optimizer


''' ACCURACY METRIC'''
def accuracy(pred_y, ground_truth_y):
    pred_y = pred_y.cpu().detach().numpy()
    pred_y = np.argmax(pred_y, axis=1)
    
    correct = pred_y - ground_truth_y.cpu().detach().numpy()
    correct_freq = np.count_nonzero(correct==0)
    total_freq = len(correct)

    accuracy_value = (correct_freq / total_freq) * 100
    return accuracy_value

EPOCH = 200
BATCH_SIZE = 128

''' TRAINING STAGE '''
def train():
    '''Training stage'''
    # Perform a forward pass bypassing our data and finding out the predicted value of y
    # Compute the loss using MSE.
    # Reset all the gradients to 0, perform a backpropagation and then, update the weights.
    for epoch in range(EPOCH):
        for batch_num in range(len(x_for_train) // BATCH_SIZE):
            x_for_train_batch = x_for_train[batch_num * BATCH_SIZE : BATCH_SIZE * (batch_num+1)]
            train_y_batch = train_y[batch_num * BATCH_SIZE : BATCH_SIZE * (batch_num+1)]

            optimizer.zero_grad()                                 # Zero gradients
            pred_y = our_model(x_for_train_batch)                       # Forward pass: Compute predicted y by passing x to the model
            # pred_y = torch.squeeze(pred_y, dim=1)                 # (128, 50) ------> (128, 1)
            # pred_y = torch.squeeze(pred_y, dim=1)                 # (128, 1)    ------> (128)
            loss = the_loss(pred_y, train_y_batch)                      # Compute and print loss
            loss.backward()                                       # perform a backward pass
            optimizer.step()                                      # update the weights.

        if (epoch %10 == 0):
            accuracy_value = accuracy(pred_y, train_y_batch)            # not really needed
            print('\033[35m' + f'epoch {epoch}, train loss {loss.item()}, train accuracy {accuracy_value}' + '\33[97m')
            test()


''' TESTING STAGE '''
def test():
    our_model.eval()
    with torch.no_grad():
        pred_y = our_model(x_for_test)                         # make predicitoin 
        pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 50) ------> (2000, 1)
        # pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 1)    ------> (2000)
        print(pred_y[1], pred_y[1027]) 
        loss = the_loss(pred_y, test_y)                        # calculate test loss - not really needed
        
        accuracy_value = accuracy(pred_y, test_y)              # to check model performance 
        print()
        print('\033[94m' + f'test loss {loss}, test accuracy {accuracy_value}' + '\33[97m')
    our_model.train()


train()
test()