#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:35:29 2020

@author: Jake
"""

# Imports
import pandas as pd
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
              'PI_Rating',
              'PI_Text',
              'YO_Rating',
              'YO_Text',
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
    print('\n--- Pilates Ratings ---\n{}'.format(df['PI_Rating'].
                                                 value_counts()))
    print('\n--- Yoga Ratings ---\n{}'.format(df['YO_Rating'].
                                              value_counts()))

# Check for duplicate respondent IDs
print('Duplicate IDs: {}'.format(df['Respondent_ID'].duplicated().sum()))

# Check basics of dataframe
print('Total Responses: {}'.format(len(df)))

# Look at ratings before and after translating text to numerical
print_ratings(df)

# Translate text ratings into numerical - can't cast to int because NaN
def text_to_num(df):
    '''
    Parameters
    ----------
    df : input dataframe
        Translates str (e.g. "Dissatisfied") to float (e.g. 1.0)

    Returns
    -------
    altered dataframe

    '''
    text_num_dict = {'I really enjoy them': 5,
                     'Very Satisfied': 5,
                     'Neutral': 3,
                     'Dissatisfied': 1}
    df['SC_Rating'] = df['SC_Rating'].apply(lambda x: text_num_dict[x]
                                            if x in text_num_dict else x)
    df['FL_Rating'] = df['FL_Rating'].apply(lambda x: text_num_dict[x]
                                            if x in text_num_dict else x)
    df['PI_Rating'] = df['PI_Rating'].apply(lambda x: text_num_dict[x]
                                            if x in text_num_dict else x)
    df['YO_Rating'] = df['YO_Rating'].apply(lambda x: text_num_dict[x]
                                            if x in text_num_dict else x)
    return df

# Check ratings after translating to numerical
df = text_to_num(df)
print_ratings(df)
print(df.describe())

# Bar plot average ratings
fig, ax = plt.subplots()
ax.bar('Step Challenge', df['SC_Rating'].mean())
ax.bar('Fit Lab', df['FL_Rating'].mean())
ax.bar('Pilates', df['PI_Rating'].mean())
ax.bar('Yoga', df['YO_Rating'].mean())
ax.grid(axis = 'y', which = 'both')

for i, v in enumerate(df[['SC_Rating', 'FL_Rating',
                          'PI_Rating', 'YO_Rating']].mean()):
    ax.text(i,
            v - .25,
            str(round(v, 2)),
            ha = 'center',
            fontsize = 10,
            fontweight = 'bold')

ax.set_title('Average Score per Class', fontsize = 15)
plt.show()

# Investigate low ratings (<=2) for Fit Lab, Pilates, Yoga
def low_scores(df, class_abbrev, score):
    '''
    Parameters
    ----------
    df: input dataframe
    class_abbrev : str, SC / FL / PI / YO
    score: int, max score to look at
        Prints out low scores and associated text.

    Returns
    -------
    None.

    '''
    rating_string = class_abbrev + '_Rating'
    text_string = class_abbrev + '_Text'
    print('{} | {}'.format(rating_string, text_string))
    low_ratings = df[df[rating_string] <= score]\
                     [[rating_string, text_string]].dropna()
    for entry in low_ratings.values:
        print('Rating: {} | Comment: {}'.format(int(entry[0]), entry[1]))
    print()

for i in ['SC', 'FL', 'PI', 'YO']:
    low_scores(df, i, 3)

# Check gym text entries
print('\nGym feedback count: {}'.format(df['Gym_Text'].notna().sum()))

# Count number of different words used in responses
def perform_LDA(df, class_abbrev, num_topics):
    '''
    Parameters
    ----------
    df: input dataframe
        Runs LDA on gym text.
        
    Returns
    -------
    None.

    '''
    text_string = class_abbrev + '_Text'
    count_vec = CountVectorizer(stop_words = 'english')
    doc_term_matrix = count_vec.fit_transform(df[text_string].
                                              values.astype('U'))
    print('Number of different words: {}'.format(doc_term_matrix.shape[1]))
    
    # Fit number of different topics
    LDA = LatentDirichletAllocation(n_components = num_topics,
                                    random_state = 42)
    LDA.fit(doc_term_matrix)
    
    # Check top 10 words in order for each topic
    for i, topic_num in enumerate(LDA.components_):
        print('\n--- Topic {} - Top 10 Words ---'.format(i + 1))
        print([count_vec.get_feature_names()[j]
               for j in LDA.components_[i].argsort()[-10:][::-1]])

# Space is concern. Look for comments on "space", "big", "large", "room"
def word_analysis(df, class_abbrev, text_list):
    '''
    Parameters
    ----------
    df : input dataframe
    class_abbrev : str, SC / FL / PI / YO
    text_list: list-like, list of words to include
        Prints comments for class_abbrev that include text_list.

    Returns
    -------
    None.

    '''
    text_string = class_abbrev + '_Text'
    text_ser = pd.Series(dtype = 'str')
    for word in text_list:
        text_ser = text_ser.append(
            df[df[text_string].str.contains(word,
                                            na = False,
                                            case = False)][text_string])
    text_ser = text_ser.drop_duplicates().reset_index(drop = True)
    print('\n--- {} Responses Which Include {} ---'.format(class_abbrev,
                                                           text_list))
    _ = [print(str(i) + '. ' + str(j)) for i, j in
         enumerate(text_ser.values, 1)]
    print('\n--- {} out of {} comments mention {} ---'.\
          format(len(text_ser), df[text_string].notna().sum(), text_list))

perform_LDA(df, 'Gym', 3)
word_analysis(df, 'Gym', ['space', 'big', 'large', 'room'])
word_analysis(df, 'Gym', ['vent', 'dirt', 'sweat', 'clean'])