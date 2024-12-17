#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# ===============================================================================
#
#           FILE: 01_partition_data.py
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS:
#        VERSION: 3.11.4
#        CREATED: 29-05-2024
#
# ===============================================================================
#    DESCRIPTION:  Filters the corpus, applies preprocessing and creates an
#                  evaluation set
#
#   DEPENDENCIES: pandas, sklearn, apply_preprocessing, polars
#
#          USAGE: python  01_partition_data.py
# ===============================================================================


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_doc import apply_preprocessing
from collections import Counter
import polars as pl


# # Load data

# In[3]:


pd.set_option("display.max_colwidth", 300)


# In[4]:


df = pd.read_csv("data/all_tweets.csv")


# In[5]:


df


# In[6]:


# transform date column to date type
df["created_at"] = pd.to_datetime(df["created_at"])

# transform party to categorical (one hot)
df["party_dummy"] = df["party"].apply(lambda label: 1 if label == "EM" else 0)


# In[7]:


df


# In[8]:


def test_tokenizer(i):
    test = df.iloc[i]["text"]
    tk = apply_preprocessing(test)

    print("Texte original :", test)
    print("Tokens :", tk)


test_tokenizer(6)


# In[9]:


def process_oov(tokens):
    preprocessed_tokens = ["<UNK>" if voc[token] < 4 else token for token in tokens]

    return preprocessed_tokens


# # Apply preprocessing

# In[2]:


# Map custom preprocessing func to the text column
df["preprocessed_text"] = df["text"].map(apply_preprocessing)


# In[11]:


# df['preprocessed_text'] = df['preprocessed_text'].map(process_oov)


# In[12]:


# unnest data for csv
df["preprocessed_text"] = df["preprocessed_text"].map(lambda tokens: " ".join(tokens))


# # Write results

# In[26]:


df.to_csv("data/all_tweets_preprocessed.csv", encoding="utf8", sep=",")


# In[27]:


# train set
train_df, test_df, train_y, test_y = train_test_split(df, df["party"], test_size=0.30)

# dev set
train_df, dev_df, train_y, dev_y = train_test_split(train_df, train_y, test_size=0.20)


# In[28]:


train_df.to_csv("data/train.csv", index=False, encoding="utf8")
test_df.to_csv("data/test.csv", index=False, encoding="utf8")
tsdev_df.to_csv("data/dev.csv", index=False, encoding="utf8")
