#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8
# ===============================================================================
#
#           FILE: 03_get_embeddings.py
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS:
#        VERSION: 3.11.4
#        CREATED: 29-05-2024
#
# ===============================================================================
#    DESCRIPTION: Creates BERT embeddings using the preprocessed text and saves
#
#   DEPENDENCIES: torch, numpy, transformers, numpy, polars
#
#          USAGE: python  03_get_embeddings.py
# ===============================================================================


# In[2]:


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import polars as pl


# # Load BERT

# In[3]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Yanzhu/bertweetfr-base")
model = AutoModelForMaskedLM.from_pretrained("Yanzhu/bertweetfr-base", output_hidden_states=True)


# In[4]:


model.config


# # Load data

# In[5]:


df = pl.read_csv("data/all_tweets_preprocessed.csv", separator=",", encoding="utf8")
train_df = pl.read_csv("data/train.csv", separator=",", encoding="utf8")
test_df = pl.read_csv("data/test.csv", separator=",", encoding="utf8")
dev_df = pl.read_csv("data/dev.csv", separator=",", encoding="utf8")


# In[6]:


df = df.with_columns(pl.col("preprocessed_text").str.split(" "))
train_df = train_df.with_columns(pl.col("preprocessed_text").str.split(" "))
test_df = test_df.with_columns(pl.col("preprocessed_text").str.split(" "))
dev_df = dev_df.with_columns(pl.col("preprocessed_text").str.split(" "))


# In[7]:


longest_sequence = len(max(df["preprocessed_text"], key=len))
print(longest_sequence)


# # Get BERT embeddings

# In[8]:


def get_bert_embeddings(texts):
    texts = texts.to_list()

    inputs = tokenizer(texts, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_layer = outputs.hidden_states[-1]  # get embeddings from last layer
    sentence_embedding = torch.mean(last_layer, dim=1).squeeze().numpy()

    return sentence_embedding


# In[9]:


# 1 == EM
# 0 == FN


# In[10]:


n_dim = model.config.hidden_size


# In[11]:


X_train, y_train = (
    train_df.select(pl.col("preprocessed_text").map_elements(get_bert_embeddings)).to_numpy(),
    train_df["party_dummy"],
)
X_test, y_test = (
    test_df.select(pl.col("preprocessed_text").map_elements(get_bert_embeddings)).to_numpy(),
    test_df["party_dummy"],
)
X_dev, y_dev = (
    dev_df.select(pl.col("preprocessed_text").map_elements(get_bert_embeddings)).to_numpy(),
    dev_df["party_dummy"],
)


# ## Reshape embeddings to 2D
# (n_rows, bert_n_dim)

# In[12]:


# unnest
X_train_flattened = [x[0] for x in X_train]
X_test_flattened = [x[0] for x in X_test]
X_dev_flattened = [x[0] for x in X_dev]


# In[13]:


X_train_np = np.vstack(X_train_flattened)
X_test_np = np.vstack(X_test_flattened)
X_dev_np = np.vstack(X_dev_flattened)


# In[2]:


# Save embeddings


# In[14]:


np.save("train_embeddings.npy", X_train_np)
np.save("test_embeddings.npy", X_test_np)
np.save("dev_embeddings.npy", X_dev_np)
