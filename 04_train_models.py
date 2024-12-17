#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: 04_train_models.py
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 29-05-2024 
#
#===============================================================================
#    DESCRIPTION: some code was generated with GPT for productivity reasons
#                 this code trains a logistic regression model, mlp and dt
#                 we evaluate using voting classifier
#    
#   DEPENDENCIES: pandas, polars, matplotlib, seaborn, numpy, scipy, sklearn
#
#          USAGE: python  04_train_models.py 
#===============================================================================


# In[2]:


from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LearningCurveDisplay, ValidationCurveDisplay, learning_curve, validation_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as stats

import numpy as np
import polars as pl
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd


# ## Load data

# In[3]:


df = pl.read_csv('data/all_tweets_preprocessed.csv')
train_df = pl.read_csv('data/train.csv')
test_df = pl.read_csv('data/test.csv')


# In[4]:


X_train, y_train = np.load('train_embeddings.npy'), train_df['party_dummy'].to_numpy()
X_test, y_test = np.load('test_embeddings.npy'), test_df['party_dummy'].to_numpy()
X_dev, y_dev = np.load('dev_embeddings.npy'), dev_df['party_dummy'].to_numpy()


# ## Vectorize texts

# In[5]:


token_pattern=r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+' #keep hashtags and @
vectorizer = CountVectorizer(token_pattern=token_pattern, stop_words=['rt', 'gt', 'http', 'amp'])

X_train_counts = vectorizer.fit_transform(train_df['preprocessed_text'].to_list())
X_test_counts = vectorizer.transform(test_df['preprocessed_text'].to_list())
X_dev_counts = vectorizer.transform(dev_df['preprocessed_text'])


# In[6]:


num_features = len(vectorizer.vocabulary_)

print("Number of features:", num_features)


# ## Logistic regression

# In[7]:


logreg = LogisticRegression(max_iter=1000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 0.7, 0.5, 0.6, 0.3],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
}
    
# Perform grid search with cross-validation
grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_train_counts, y_train)

## Scores
best_params_lg = grid_search.best_params_
best_score_lg = grid_search.best_score_

print("Best Parameters:", best_params_lg)
print("Best Cross-Validation Score:", best_score_lg)

#### BEST MODEL ###
best_logreg = LogisticRegression(**best_params_lg)
best_logreg.fit(X_train_counts, y_train)

# Evaluate on the test set as well
y_test_pred = best_logreg.predict(X_test_counts)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=['EM','FN']))


# ### Linearity check

# In[8]:


logits = best_logreg.decision_function(X_train_counts)

# Compute the residuals
residuals = y_train - logits

# Generate QQ plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('')
plt.xlabel('Quantiles théoriques')
plt.ylabel('Quantiles des résidus')
plt.savefig('paper/residuals.pdf', format="pdf")
plt.show()


# In[9]:


## get real labels
ground_truth = np.ones(X_train_counts.shape[0], dtype=int)

## use LOF for detecting outliers
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

y_pred = clf.fit_predict(X_train_counts)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_


# In[10]:


threshold = -2.5

# identify outlier indices based on the threshold
outlier_indices = np.where(X_scores < threshold)[0]

# visualize the outlier indices
print("Outlier Indices:", outlier_indices)


# In[11]:


## filter train and check if higher performance without outliers

# train_df = train_df.to_pandas().drop(outlier_indices)
# y_train = train_df['party_dummy'].to_numpy()


# ### ROC, Validation and Learning curve

# In[12]:


y_test_prob = best_logreg.predict_proba(X_test_counts)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Aire de la courbe ROC (aire = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('')
plt.legend(loc="lower right")

plt.savefig('paper/courbeROC.pdf', format="pdf")
plt.show()


# In[13]:


## Load C parameter values
param_name, param_range = "C", np.logspace(-5, 4, 3)

train_scores, test_scores = validation_curve(
    best_logreg, X_train_counts, y_train, param_name=param_name, param_range=param_range, cv=5
)

## valid curve for C
valid_display = ValidationCurveDisplay(
    param_name=param_name, param_range=param_range,
    train_scores=train_scores, test_scores=test_scores, score_name="Score"
)
## display
valid_display.plot()
plt.savefig('paper/C_parameter.pdf')


# In[14]:


### learning curve
train_sizes, train_scores, test_scores = learning_curve(
    best_logreg, X_train_counts, y_train, cv=10
)

## display
test_display = LearningCurveDisplay(
    train_scores=train_scores, test_scores=test_scores, train_sizes=train_sizes, score_name="Score"
)
test_display.plot()
plt.xlabel("Nombre d'échantillons dans le train set")

plt.savefig('paper/learning_curve.pdf')


# ### Most important variables

# In[15]:


feature_names = vectorizer.get_feature_names_out()

# Get the coefficients from the best logistic regression model
coefficients = best_logreg.coef_[0]

# Create a DataFrame to neatly display feature names and their corresponding coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Separate the positive and negative coefficients
positive_features = feature_importance.sort_values(by='Coefficient', ascending=False).head(10)
negative_features = feature_importance.sort_values(by='Coefficient', ascending=True).head(10)

print("Most Important Positive Features (increase likelihood of target class):")
print(positive_features)

print("\nMost Important Negative Features (decrease likelihood of target class):")
print(negative_features)


# ## Decision Trees

# In[16]:


#### DT TRAINING ####

dt = DecisionTreeClassifier(random_state=0)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(2,10)),
    'min_samples_split': [2,4,5,8,10],
    'min_samples_leaf': [2,3,5,7],
    'ccp_alpha': [0.03, 0.1, 0.5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, 
                           cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train_counts, y_train)

# Best parameters and best score
best_params_dt = grid_search.best_params_
best_score_dt = grid_search.best_score_

print("Best Parameters:", best_params_dt)
print("Best Cross-Validation Score:", best_score_dt)

best_dt = DecisionTreeClassifier(**best_params_dt)
best_dt.fit(X_train_counts, y_train)

# Evaluate on the test set as well
y_test_pred_dt = best_dt.predict(X_test_counts)
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)
print("Test set Accuracy:", test_accuracy_dt)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_dt, target_names=['EM','FN']))


# ## NN

# In[ ]:


mlp_clf = MLPClassifier(random_state=0, learning_rate='adaptive', max_iter=500)

param_grid_mlp = {
    'hidden_layer_sizes': [(1,3),(2,3), (3,4)],
    'activation': ['logistic'],
    'solver': ['adam'],
    'alpha': [2e-5, 1e-3, 0.0001, 0.001, 0.025, 0.1]
}


grid_search_mlp = GridSearchCV(mlp_clf, param_grid_mlp, cv=5, scoring='accuracy')
grid_search_mlp.fit(X_train_counts, y_train)

# Best parameters and best score
best_params_mlp = grid_search_mlp.best_params_
best_score_mlp = grid_search_mlp.best_score_

print("Best Parameters:", best_params_mlp)
print("Best Cross-Validation Score:", best_score_mlp)


# In[ ]:


best_mlp = MLPClassifier(**best_params_mlp, max_iter=1000)
best_mlp.fit(X_train_counts, y_train)


# Evaluate on the test set as well
y_test_pred = best_mlp.predict(X_test_counts)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=['EM','FN']))


# In[ ]:


### BERT ###
mlp_bert = MLPClassifier(activation='logistic', alpha= 0.02, hidden_layer_sizes=(1,3), max_iter=1000, learning_rate='adaptive')
mlp_bert.fit(X_train, y_train)


# Evaluate on the test set as well
y_test_pred = mlp_bert.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=['EM','FN'])))


# # Voting classifiers

# In[ ]:


eclf1 = VotingClassifier(estimators=[
    ('logreg', best_logreg),
    ('dt', best_dt),
    ('mlp', best_mlp),
], voting='soft')

# Train the voting classifier
eclf1.fit(X_train_counts, y_train)

# Predict on the test set
y_pred = eclf1.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names = ['EM', 'FN'])

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

error: cannot format -: Cannot parse: 364:103: print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=['EM','FN'])))
