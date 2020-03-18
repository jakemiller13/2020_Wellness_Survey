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

# Check for duplicate respondent IDs
print('Duplicate IDs: {}'.format(df['Respondent_ID'].duplicated().sum()))

# Look at ratings
print('\n--- Step Challenge Ratings ---\n{}'.format(df['SC_Rating'].
                                                    value_counts()))
print('\n--- Fit Lab Ratings ---\n{}'.format(df['FL_Rating'].
                                             value_counts()))
print('\n--- Pilates Ratings ---\n{}'.format(df['Pi_Rating'].
                                             value_counts()))
print('\n--- Yoga Ratings ---\n{}'.format(df['Yo_Rating'].
                                          value_counts()))

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

# Bar plot average ratings
fig, ax = plt.subplots(figsize = (10, 10))
ax.bar('Step Challenge', df['SC_Rating'].mean())
ax.bar('Fit Lab', df['FL_Rating'].mean())
ax.bar('Pilates', df['Pi_Rating'].mean())
ax.bar('Yoga', df['Yo_Rating'].mean())
ax.grid(axis = 'y', which = 'both')

for i, v in enumerate(df[['SC_Rating', 'FL_Rating',
                          'Pi_Rating', 'Yo_Rating']].mean()):
    ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

plt.show()
