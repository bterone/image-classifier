
# coding: utf-8

# # Building a Chatbot

# In this project, we will build a chatbot using conversations from Cornell University's [Movie Dialogue Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The main features of our model are LSTM cells, a bidirectional dynamic RNN, and decoders with attention. 
# 
# The conversations will be cleaned rather extensively to help the model to produce better responses. As part of the cleaning process, punctuation will be removed, rare words will be replaced with "UNK" (our "unknown" token), longer sentences will not be used, and all letters will be in the lowercase. 
# 
# With a larger amount of data, it would be more practical to keep features, such as punctuation. However, I am using FloydHub's GPU services and I don't want to get carried away with too training for too long.

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__


# Most of the code to load the data is courtesy of https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py.

# ### Inspect and Load the Data

# In[2]:

# Load the data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


# In[3]:

# The sentences that we will be using to train our model.
print(lines[:10])


# In[4]:

# The sentences' ids, which will be processed to become our input and target data.
print(conv_lines[:10])

# In[5]:

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# In[6]:

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))


# In[7]:

print(convs[:10])


# In[8]:

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# In[9]:

# Check if we have loaded the data correctly
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()


# In[10]:

# Compare lengths of questions and answers
print(len(questions))
print(len(answers))

# In[11]:

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# In[12]:

# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))


# In[13]:

# Take a look at some of the data to ensure that it has been cleaned well.
limit = 0
for i in range(limit, limit+5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()


# In[14]:

# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# In[15]:

lengths.describe()

# In[16]:

print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))


# In[17]:

# Remove questions and answers that are shorter than 2 words and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1


# In[18]:

# Compare the number of lines we will use with the total number of lines.
print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))

# In[19]:

# Create a dictionary for the frequency of the vocabulary
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
            
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# In[20]:

# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1


# In[21]:

print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)

# In[22]:

# In case we want to use a different vocabulary sizes for the source and target text, 
# we can set different threshold values.
# Nonetheless, we will create dictionaries to provide a unique integer for each word.
questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1
        
answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1

