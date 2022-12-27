import torch
import numpy as np
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from preprocessing import *
from building_vocabulary import *
from preparing_data import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''PREPROCESS STAGE'''

# prepare the datasets 
all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels, classes = accessing_data()
train_x, train_y = train_dataset(all_positive_tweets, all_negative_tweets)
test_x, test_y = test_dataset(all_positive_tweets, all_negative_tweets)

# process training data
processed_train_x = process_data(train_x)
processed_test_x = process_data(test_x)

# build vocabulary
freqs = build_freqs(train_x, train_y) 

# convert to tensor
x_for_train, train_y = convert_to_tensors(processed_train_x, train_y, freqs, device)
x_for_test, test_y = convert_to_tensors(processed_test_x, test_y, freqs, device)


''' OUR MODEL '''
class LinearRegressionModel(torch.nn.Module):            # Initializing our model.

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)    # input_dim=2 and Output_dim = 1
 
    def forward(self, x):                                 # Declaring the forward pass.
        y_pred = self.linear(x)
        return y_pred

our_model = LinearRegressionModel().to(device=device)               # object of this model     
the_loss = torch.nn.MSELoss()                                       # the loss criteria
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.001)     # optimizer


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
# Perform a forward pass bypassing our data and finding out the predicted value of y
def train():
    EPOCH = 10000
    for epoch in range(EPOCH):
        optimizer.zero_grad()                                 # Zero gradients
        pred_y = our_model(x_for_train)                       # Forward pass: Compute predicted y by passing x to the model
        pred_y = torch.squeeze(pred_y, dim=1)                 # (8000, 1, 2) ------> (8000, 1)
        pred_y = torch.squeeze(pred_y, dim=1)                 # (8000, 1)    ------> (8000)
        loss = the_loss(pred_y, train_y)                      # Compute loss using MSE
        loss.backward()                                       # perform a backward pass
        optimizer.step()                                      # update the weights.

        if (epoch %100 == 0):
            accuracy_value = accuracy(pred_y, train_y)            # not really needed
            # print()
            print('\033[35m' + f'epoch {epoch}, train loss {loss.item()}, train accuracy {accuracy_value}'  + '\33[97m')
            test()


''' TESTING STAGE '''
def test():
    with torch.no_grad():
        pred_y = our_model(x_for_test)                         # make prediction 
        pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 1, 2) ------> (2000, 1)
        pred_y = torch.squeeze(pred_y, dim=1)                  # (2000, 1)    ------> (2000)
        print(pred_y[1], pred_y[1027])
        loss = the_loss(pred_y, test_y)                        # calculate test loss - not really needed

        accuracy_value = accuracy(pred_y, test_y)              # to check model performance 


        print()
        
        print('\033[94m' + f'test loss {loss}, test accuracy {accuracy_value}'  + '\33[97m')


train()
test()