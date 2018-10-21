from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split


data = Dataset.load_builtin('jester')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset = data.build_full_trainset()

# We'll use the famous SVD algorithm.
algo = SVD()


# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
print("fit")
print(algo.predict(str(1),str(3)))
# predictions = algo.test(testset)

# # Then compute RMSE
# print(accuracy.rmse(predictions))

import pandas as pd

test_df = pd.read_csv("test.csv")
print(test_df.head(5))

def label_rating(row):
    return algo.predict(str(row['user_id']),str(row['joke_id']))[3]

test_df['Rating'] = test_df.apply (lambda row: label_rating (row),axis=1)

print(test_df.head(5))

sub_df = test_df.drop('user_id',1)

sub_df = sub_df.drop('joke_id',1)

sub_df.to_csv('out1.csv',index=False)


