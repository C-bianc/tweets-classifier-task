#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# ===============================================================================
#
#           FILE: 02_get_statistics.py
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS:
#        VERSION: 3.11.4
#        CREATED: 29-05-2024
#
# ===============================================================================
#    DESCRIPTION: Provides descriptive statistics about the dataset
#
#   DEPENDENCIES: pandas, polars, matplotlib, seaborn, scipy
#
#          USAGE: python  02_get_statistics.py
# ===============================================================================


# In[2]:


import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[3]:


pl.Config.set_tbl_rows(20)


# In[4]:


# Load data

df = pl.read_csv("data/all_tweets_preprocessed.csv")
train_df = pl.read_csv("data/train.csv", separator=",", encoding="utf8")
test_df = pl.read_csv("data/test.csv", separator=",", encoding="utf8")
dev_df = pl.read_csv("data/dev.csv", separator=",", encoding="utf8")


# In[5]:


# split text into words

df = df.with_columns(pl.col("preprocessed_text").str.split(" "))
df = df.with_columns(pl.col("preprocessed_text").list.eval(pl.element().filter(pl.element() != "#")))  # remove empty #
df = df.with_columns(pl.col("preprocessed_text").list.eval(pl.element().filter(pl.element() != "")))  # remove empty #


# In[6]:


# get all words
voc = (
    df.select(pl.col("preprocessed_text").flatten())
    .group_by("preprocessed_text")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
voc = voc.with_columns((pl.col("count") / voc.height).alias("proportion_corpus"))


# In[7]:


print(voc)


# In[8]:


print("unique words: ", voc.height)
print("all words: ", voc["count"].sum())


# In[9]:


least_frequent_counts = voc.filter(pl.col("count") == 1)
print("% Count(x) == 1 : ", least_frequent_counts.height / voc.height)


# In[10]:


# unique_hashtags = voc.filter((pl.col('count')==1) & (pl.col('preprocessed_text').str.starts_with("#")))
# unique_mentions = voc.filter((pl.col('count')==1) & (pl.col('preprocessed_text').str.starts_with("@")))


# In[11]:


def get_propr_classes(df, df_type):
    corpus_party = df["party"].to_pandas().value_counts()

    print(df_type, corpus_party)
    print("\nEM propr: ", corpus_party["EM"] / len(df) * 100)
    print("FN propr: ", corpus_party["FN"] / len(df) * 100)
    print()


get_propr_classes(df, "corpus")
get_propr_classes(train_df, "train")
get_propr_classes(test_df, "test")
get_propr_classes(dev_df, "dev")


# In[12]:


### WORD TYPE ###
mentions = voc.filter(pl.col("preprocessed_text").str.starts_with("@"))
mentions = mentions.with_columns((pl.col("count") / mentions["count"].sum()).alias("proportion"))
print("Nombre total : ", mentions.get_column("count").sum())
print("Proportion des mentions dans le corpus : ", mentions["count"].sum() / voc["count"].sum())
print(mentions)

hashtags = voc.filter(pl.col("preprocessed_text").str.starts_with("#"))
hashtags = hashtags.with_columns((pl.col("count") / hashtags["count"].sum()).alias("proportion"))
print("Nombre total : ", hashtags.get_column("count").sum())
print("Proportion des hashtags: ", hashtags["count"].sum() / voc["count"].sum())
print(hashtags)

other = voc.filter(
    (~pl.col("preprocessed_text").str.starts_with("@")) & (~pl.col("preprocessed_text").str.starts_with("#"))
)
other = other.with_columns((pl.col("count") / other["count"].sum()).alias("proportion"))
print("Proportion du reste: ", other["count"].sum() / voc["count"].sum())
print(other)


# In[13]:


print("Num of tweets : ", df.height)
print("voc size :      ", voc.height)
print("% mentions :    ", mentions.height / voc.height)
print("% hashtags :    ", hashtags.height / voc.height)
print("% autres   :    ", other.height / voc.height)


# In[14]:


def plot_freq(data, x_axis, y_axis, name, x_label):
    sns.barplot(data[:20], x=x_axis, y=y_axis, hue=y_axis, palette=sns.color_palette("crest_r", 20))
    plt.xticks(rotation=80)

    plt.tight_layout()
    plt.xlabel(x_label, labelpad=10)
    plt.ylabel("Proportion")

    plt.savefig(f"paper/{name}.pdf", format="pdf")
    plt.show()


plot_freq(
    voc.filter(~pl.col("preprocessed_text").str.contains("<UNK>")),
    "proportion_corpus",
    "preprocessed_text",
    "word_freq_top20",
    "Mots",
)
plot_freq(mentions, "proportion", "preprocessed_text", "mentions", "Mentions")
plot_freq(hashtags, "proportion", "preprocessed_text", "hashtags", "Mots-dièse")


# In[15]:


# check tweet length

length_train = train_df.select(pl.col("text").str.len_bytes().alias("text_len"))
length_test = test_df.select(pl.col("text").str.len_bytes().alias("text_len"))


def plot_tweet_len():
    print("Moyenne des tweets pour train: ", length_train.mean().item())
    print("Ecart-type des tweets pour train: ", length_train.std().item())
    print()
    print("Moyenne des tweets pour test: ", length_test.mean().item())
    print("Ecart-type des tweets pour test: ", length_test.std().item())

    plt.hist(length_train, bins=20, label="train_tweets", color="lightsteelblue")
    plt.hist(length_test, bins=20, label="test_tweets", color="navy")
    plt.legend(title="Split", labels=["Train", "Test"])
    plt.ylabel("Fréquence")
    plt.xlabel("Longueur des tweets")

    plt.savefig("paper/tweets_len.pdf", format="pdf")
    plt.show()


plot_tweet_len()


# In[16]:


# check distributions
ks_statistic, ks_p_value = stats.ks_2samp(length_train["text_len"].to_list(), length_test["text_len"].to_list())
print(f"K-S Test Statistic: {ks_statistic}, p-value: {ks_p_value}")
