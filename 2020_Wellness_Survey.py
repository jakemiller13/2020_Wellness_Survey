#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:35:29 2020

@author: Jake
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Read in data, drop unneccessary columns
df = pd.read_excel('2020_Wellness_Survey.xlsx', skiprows = [1])
df = df.drop(['Collector ID', 'IP Address', 'Email Address',
              'First Name', 'Last Name', 'Custom Data 1'], axis = 1)

# Rename columns for easier printing
og_cols = df.columns
df.columns = ['Respondent_ID',
              'Start_Date',
              'End_Date',
              'SC_Rating',
              'SC_Text',
              'FL_Rating',
              'FL_Text',
              'Pi_Rating',
              'Pi_Text',
              'Yo_Rating',
              'Yo_Text',
              'Gym_Use',
              'Gym_Text']

def print_ratings(df):
    '''
    Prints out ratings for each class and step challenge
    '''
    print('\n--- Step Challenge Ratings ---\n{}'.format(df['SC_Rating'].
                                                        value_counts()))
    print('\n--- Fit Lab Ratings ---\n{}'.format(df['FL_Rating'].
                                                 value_counts()))
    print('\n--- Pilates Ratings ---\n{}'.format(df['Pi_Rating'].
                                                 value_counts()))
    print('\n--- Yoga Ratings ---\n{}'.format(df['Yo_Rating'].
                                              value_counts()))

# Check for duplicate respondent IDs
print('Duplicate IDs: {}'.format(df['Respondent_ID'].duplicated().sum()))

# Check basics of dataframe
print('Total Responses: {}'.format(len(df)))

# Look at ratings before and after translating text
print_ratings(df)

# Translate text ratings into numerical - can't cast to int because NaN
text_to_num = {'I really enjoy them': 5,
               'Very Satisfied': 5,
               'Neutral': 3,
               'Dissatisfied': 1}
df['SC_Rating'] = df['SC_Rating'].apply(lambda x: text_to_num[x]
                                        if x in text_to_num else x)
df['FL_Rating'] = df['FL_Rating'].apply(lambda x: text_to_num[x]
                                        if x in text_to_num else x)
df['Pi_Rating'] = df['Pi_Rating'].apply(lambda x: text_to_num[x]
                                        if x in text_to_num else x)
df['Yo_Rating'] = df['Yo_Rating'].apply(lambda x: text_to_num[x]
                                        if x in text_to_num else x)
print_ratings(df)

# Bar plot average ratings
fig, ax = plt.subplots(figsize = (10, 10))
ax.bar('Step Challenge', df['SC_Rating'].mean())
ax.bar('Fit Lab', df['FL_Rating'].mean())
ax.bar('Pilates', df['Pi_Rating'].mean())
ax.bar('Yoga', df['Yo_Rating'].mean())
ax.grid(axis = 'y', which = 'both')

for i, v in enumerate(df[['SC_Rating', 'FL_Rating',
                          'Pi_Rating', 'Yo_Rating']].mean()):
    ax.text(i,
            v + .05,
            str(round(v, 2)),
            ha = 'center',
            fontsize = 15,
            fontweight = 'bold')

ax.set_title('Average Score per Class', fontsize = 20)
plt.show()

# Check gym text entries
print('Gym feedback count: {}'.format(df['Gym_Text'].notna().sum()))

# Count number of different words used in responses
count_vec = CountVectorizer(stop_words = 'english')
doc_term_matrix = count_vec.fit_transform(df['Gym_Text'].values.astype('U'))
print('Number of different words: {}'.format(doc_term_matrix.shape[1]))

# Fit number of different topics
LDA = LatentDirichletAllocation(n_components = 3, random_state = 42)
LDA.fit(doc_term_matrix)

# Check top 10 words for each topic
for i, topic_num in enumerate(LDA.components_):
    print('\n--- Topic {} - Top 10 Words ---'.format(i + 1))
    print([count_vec.get_feature_names()[j]
           for j in LDA.components_[i].argsort()[-10:]])