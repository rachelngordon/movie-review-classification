 # Implement your own vectorizer to convert the text of the input documents into vectors that can be
# used to train a scikit-learn classifier

import sklearn
import pandas as pd
import numpy as np

# open the data files from github
neg_reviews = list(open('cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.neg', 'r'))
pos_reviews = list(open('cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.pos', 'r'))

# clean the text by removing newline characters and punctuation
def clean_text(old_list):

    new_list = []
    punctuation = ['[', ']', '"', ',', '.', '(', ')', '?', '!', '&', '--']

    for review in old_list:
        review = review.rstrip('\n')
        review = review.rstrip(' .')
        review = review.rstrip(' ')
        words = review.split(' ')
        for i in range(0, len(words)):
            words[i] = words[i].lstrip("'")
            words[i] = words[i].rstrip("'")
            words[i] = words[i].lstrip('/')
            words[i] = words[i].rstrip('/')
            for item in punctuation:
                words[i] = words[i].replace(item, '')
        review = ' '.join(words)
        review = review.replace('  ', ' ')
        new_list.append(review)

    return new_list

# clean the text and randomly shuffle the reviews
neg_reviews = clean_text(neg_reviews)
pos_reviews = clean_text(pos_reviews)

# split negative and positive reviews into training, development, and test sets
num_neg = len(neg_reviews)
num_pos = len(pos_reviews)
num_negtrain = int(0.7*num_neg)
num_postrain = int(0.7*num_pos)
negdev_ind = num_negtrain + int(0.15*num_neg)
posdev_ind = num_postrain + int(0.15*num_pos)
neg_train = neg_reviews[0:num_negtrain]
pos_train = pos_reviews[0:num_postrain]
neg_dev = neg_reviews[num_negtrain:negdev_ind]
pos_dev = pos_reviews[num_postrain:posdev_ind]
neg_test = neg_reviews[negdev_ind:]
pos_test = pos_reviews[posdev_ind:]

def set_features(training_data):

    # create a list of all features in the corpus
    features = []
    for review in training_data:
        tokens = review.split(' ')
        for token in tokens:
            if token not in features:
                features.append(token)

    # discard features that occur in fewer than n and more than m examples
    n = 14
    m = 5000
    for feature in features:
        if str(training_data).count(feature) < n:
            features.remove(feature)
        elif str(training_data).count(feature) > m:
            features.remove(feature)

    return features 

# combine the negative and positive training data and set the target values
training_data = neg_train + pos_train
target_names = ['negative', 'positive']
neg_target = len(neg_train)*[0]
pos_target = len(pos_train)*[1]
train_targets = neg_target + pos_target
dev_set = neg_dev + pos_dev

# set the features for the model using both the training and development sets
train_features = set_features(training_data) 
dev_features = set_features(dev_set)

features = train_features
for feature in dev_features:
    if feature not in features:
        features.append(feature)

# implement your own vectorizer that converts the training data into a numpy array
def count_vectorizer(training_data):

    global features
  
    # create vectors of the counts and tfidf values of each feature in each document
    count_vectors_list = []

    for review in training_data:

        # counts of each feature
        count_vector = []
        for feature in features:
            count_vector.append(review.count(feature))
        count_vectors_list.append(count_vector)
     
    # create a numpy array of the count vectors for each training example
    count_vectors = np.array(count_vectors_list, np.int32)


    return count_vectors


# train a sci-kit learn classifier and use the development set to select the hyper-parameters
from sklearn.linear_model import LinearRegression, LogisticRegression

# train the initial model 
train_vectors = count_vectorizer(training_data)
#logistic_clf = LogisticRegression(max_iter = 1000).fit(train_vectors, train_targets)

# evaluate performance and select hyperparameters using development set
dev_targets = len(neg_dev)*[0] + len(pos_dev)*[1]
dev_vectors = count_vectorizer(dev_set)
#print(logistic_clf.score(dev_vectors, dev_targets))

# hyperparameter selection
# n = 5, m = 5000, score: 0.7428035043804756
# n = 10, m = 5000, score: 0.7478097622027534
# n = 20, m = 5000, score: 0.7459324155193993
# n = 15, m = 5000, score: 0.7521902377972466
# n = 13, m = 5000, score: 0.7496871088861077
# n = 17, m = 5000, score: 0.7471839799749687
# n = 14, m = 5000, score: 0.753441802252816
# n = 14, m = 6000, score: 0.7528160200250313
# n = 14, m = 4000, score: 0.7515644555694618
# n = 14, m = 5500, score: 0.7528160200250313
# n = 14, m = 5250, score: 0.7528160200250313
# n = 14, m = 5100, score: 0.753441802252816
# n = 14, m = 4500, score: 0.753441802252816
# selection: n = 14 and m = 5000

# combine the training and development sets and vectorize
final_training = training_data + dev_set
final_targets = train_targets + dev_targets
count_vectors = count_vectorizer(final_training)

# retrain the model using both the training and development sets
#clf = LogisticRegression(max_iter = 10000).fit(count_vectors, final_targets)

# test the model performance on the test set
test_data = neg_test + pos_test
test_targets = len(neg_test)*[0] + len(pos_test)*[1]
test_vectors = count_vectorizer(test_data)
#print(clf.score(test_vectors, test_targets))
# performance: 0.7540574282147315


# Use scikit-learn CountVectorizer to generate the input vectors, select the best hyper-parameters 
# using the development set, and finally evaluate your best model using the test set. 
from sklearn.feature_extraction.text import CountVectorizer

# vectorize the training data
count_vect = CountVectorizer(max_df = 4500, min_df = 4)
train_counts = count_vect.fit_transform(training_data)

# fit a logisitic regression model to the training data
#text_clf = LogisticRegression(max_iter = 10000).fit(train_counts, train_targets)

# vectorize and evaluate the model on the development set
vectorizer = CountVectorizer(max_df = 4500, min_df = 4, vocabulary = count_vect.vocabulary_)
dev_counts = vectorizer.fit_transform(dev_set)
#print(text_clf.score(dev_counts, dev_targets))

# testing development set accuracy according to hyperparameters
# max_df = 5000, min_df = 5, score: 0.7546933667083855
# max_df = 4000, min_df = 5, score: 0.753441802252816
# max_df = 6000, min_df = 5, score: 0.7546933667083855
# max_df = 3000, min_df = 5, score: 0.7528160200250313
# max_df = 4500, min_df = 5, score: 0.7546933667083855
# max_df = 4500, min_df = 15, score: 0.7227784730913642
# max_df = 4500, min_df = 10, score: 0.727784730913642
# max_df = 4500, min_df = 7, score: 0.7365456821026283
# max_df = 4500, min_df = 6, score: 0.7446808510638298
# max_df = 4500, min_df = 4, score: 0.7521902377972466
# selection: max_df = 4500, min_df = 5

# retrain the model using both the training and development sets
final_train_vectors = vectorizer.fit_transform(final_training)
final_clf = LogisticRegression(max_iter = 10000).fit(final_train_vectors, final_targets)

# test the model performance on the test set
test_counts = vectorizer.fit_transform(test_data)
print(final_clf.score(test_counts, test_targets))
# performance: 0.7409488139825219


