import numpy as np
import math

from preparing_data import *
from preprocessing import process_tweet
from building_vocabulary import build_freqs

# Step 0: Collect and annotate the corpus (positive tweet datset and negative tweet dataset)
all_positive_tweets, all_negative_tweets, all_tweets, all_positive_labels, all_negative_labels, all_labels, classes = accessing_data()

# print("\nall positive tweets 0: \n" + '\33[32m' + all_positive_tweets[0]+ "\n" + '\33[97m')
train_x, train_y = train_dataset(all_positive_tweets, all_negative_tweets)


# Step 1: Preprocessing step

# Step 2: Word count: computing the vocabulary of each word in class freq(word, class)
freqs = build_freqs(train_x, train_y)

# make a frequency table that takes {(word, class): freq} and returns {word : [neg_count, pos_count]}

def freq_table(freqs):
    table = {}
    # i, j = 0, 0
    for key, value in freqs.items():
        table_value_freq = [0, 0]       # class_freq[0] = neg  and class_freq[1] = pos
        if key[0] in table:
            old_values = table[key[0]]
            old_values[int(key[1])] = value
            table[key[0]] = old_values
        
        else:                           # if key[1] is positive
            table_value_freq = [0,0]
            table_value_freq[int(key[1])] = value
            table[key[0]] = table_value_freq
            
    return table            # {word : [neg_count, pos_count]}
           
voc_freq_table = freq_table(freqs)

class_count = np.array(list(voc_freq_table.values()))
total_class_count = np.sum(class_count, axis=0)

total_pos_count = total_class_count[1]
total_neg_count = total_class_count[0]
V = len(voc_freq_table)

# Step 3: p(w|class) = (freq(w, class) + 1) / N_class + v 
    # takes {word : [neg_count, pos_count]} ------------> {word : [prob_neg, prob_pos]} 
def conditional_probability(table): 

    for word, value in voc_freq_table.items():
        neg_count, pos_count = value[0], value[1]

        # negative
        numerator_neg = neg_count + 1
        denomenator_neg = total_neg_count + V
        prob_neg = numerator_neg / denomenator_neg

        # positive
        numerator_pos = pos_count + 1
        denomenator_pos = total_pos_count + V
        prob_pos = numerator_pos / denomenator_pos
        
        table[word] = [prob_neg, prob_pos]

    return table

prob_table = conditional_probability(voc_freq_table)

# #### Temp

# words = np.array(list(prob_table.keys()))
# probs = np.array(list(prob_table.values()))
# probs_neg_sorted = np.argsort(probs[:, 0])
# negative_words_increasing = words[probs_neg_sorted]

# # :-(  :(  :d :-) :)
# print('')

# ##### Temp

check = np.array(list(prob_table.values()))
print(np.sum(check, axis=0)) # the sum should be 1

    
# Step 4: get Lambda - ration = p(w|pos)/p(w|neg) lambda(w) = log(ration)
    # takes {word : [prob_neg, prob_pos]} ------------> {word : [prob_neg, prob_pos, lambda]}
def add_lambda_to_prob_table(table):

    for word, value in voc_freq_table.items():
        prob_neg, prob_pos = value[0], value[1]
        ratio = prob_pos / prob_neg
        lambda_w = math.log(ratio)

        table[word] = [prob_neg, prob_pos, lambda_w]

    return table 

prob_table_with_lamda = add_lambda_to_prob_table(prob_table)

# Step 5: get log prior log_prior = log(D_pos / D_neg)    ......... D_pos= number of pos tweets and D_neg = no of neg tweets
d_pos = len(all_positive_tweets)
d_neg = len(all_negative_tweets)
log_prior = math.log(d_pos / d_neg)




# step 2: look up each words in the log likelihood table
def score(testing_tweet):
    # process the testing tweet
    processed_testing_tweet = process_tweet(testing_tweet, stem=True)
    log_likelihood = 0
    for i in range (len(processed_testing_tweet)):
        word = processed_testing_tweet[i]
        if word in prob_table_with_lamda:
            value = prob_table_with_lamda[word]
            log_likelihood += value[2]
        else:
            log_likelihood += 0 

    score = log_likelihood + log_prior
    return score
        
def eval(score):
    if score > 0:
        pred = 1
        print(f"Prediction = {pred} Thus, The tweet has a positive sentiment")
    else: 
        pred = 0
        print(f"Prediction = {pred} Thus, The tweet has a negative sentiment")


testing_tweet = "I passed the NLP interview"
score_test_tweet = score(testing_tweet)
eval(score_test_tweet)


'''TESTING STEP'''
# score = predict(x_val, Lamda, log_prior)

test_x, test_y = test_dataset(all_positive_tweets, all_negative_tweets)


def predict(test_x):
    pred = []
    for test_tweet in test_x:
        test_score = score(test_tweet)
        # print(test_score)
        pred.append(test_score)
    
    for i in range(len(pred)):
        if pred[i] > 0:
            pred[i] = 1
        else:
            pred[i] = 0
    # print(pred)
    return pred

def accuracy(test_x, test_y):
    pred_m = predict(test_x)

    final_comp = []
    for i in range(len(pred_m)):
        if pred_m[i] == test_y[i]:
            final_comp.append(1)
        else:
            final_comp.append(0)
    return final_comp

# calculating accuracy 
m = len(test_x)
final_comp = accuracy(test_x, test_y)
sum_final_comp = sum(final_comp)
print(sum_final_comp)
accuracy = (1 / m) * sum_final_comp

print(accuracy)

tweet_sample = 1
testing_tweet = test_x[tweet_sample]
testing_tweet_label = test_y[tweet_sample]
print(f"Tweet: {testing_tweet} \n Ground Truth = {testing_tweet_label}")
score_test_tweet = score(testing_tweet)
print(score_test_tweet)




         
