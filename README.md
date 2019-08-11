# Recommender for CarMusic dataset

Example of context-aware recommender model for the CarMusic dataset.
Author: Oskar Jarczyk <oskar.jarczyk at pja.edu.pl>

# Table of contents

[TOC]

# Introduction

This is an alternate solution for music recommendation as originally described 
in the GitHub repository [irecsys/CARSKit](https://github.com/irecsys/CARSKit). I present here
possible ways to deal with a context-aware dataset to recommend music depending
on external conditions like: mood, weather conditions, type of road, etc. 

## Environment

Tested on Ubuntu `18.04 LTS`, standard laptop equipped with `i7` and `32GB` of RAM.

## Requirements

Requires Python `3.6+`. You also need to install required packages from PIP as listed in the `requirements.txt` file

```shell script
pip install -r requirements.txt
```

Depending on privileges, you may need to use the `--user` flag together with `pip`. 

# Project description

## Car music project

### Data

Data is downloadable from here:
 https://github.com/irecsys/CARSKit/blob/master/context-aware_data_sets/Music_InCarMusic.zip

#### dataset.csv

This is the transformed dataset saved to a CSV file. Generated by 'CarMusic exploratory analysis.ipynb'.

#### dataset_aggr.csv

This is the transformed and aggregated dataset saved to a CSV file. Generated by 'CarMusic exploratory analysis.ipynb'.

### Success metric

In true recommender systems (e.g. collaborative and/or hybrid filtering), we use **Precision** and **Recall** at K=10.
For classic machine learning solution (were we have a balanced 1/0 label for the ground truth), we use **F1-score**, 
which combines precision and recall in a weighted manner.  
Let's take a look at the definition of the *recall at k*: 
> Measure the recall at k metric for a model: the number of positive items in the first k positions
> of the ranked list of results divided by the number of positive items in the test period. 
>A perfect score is 1.0.

And the definition of the *precision at k*:
> Measure the precision at k metric for a model: the fraction of known positives in the first k positions 
>of the ranked list of results. A perfect score is 1.0.  

Both precision and recall are therefore based on an understanding and measure of relevance. F1 combines both of them.

### Analytical stage

#### Exploratory analysis

##### Rationale

**[CarMusic exploratory analysis.ipynb](https://bitbucket.org/oskar-j/scalac/src/master/Music_InCarMusic/CarMusic%20exploratory%20analysis.ipynb)** jupyter notebook 
holds exploratory analysis of the Car music datasets. 
It applies data cleaning, feature engineering and some simple aggregations. 
Data is verified for null values, correlations and type of distributions. 
We made some unsupervised data clustering 
to find naturally occurring segments of users. We tried some simple apriori
algorithm to see if it's possible to guess by the law of co-occurrence whether there are songs that are
listened to together. At the end, we save two kinds of processed datasets to a CSV file, one of them is a 
raw un-aggregated set, second one is a dataset aggregated by pair of userId - itemId. 

##### Findings 

Most of columns have a huge number of NaN values. Fortunately, there are almost no correlated columns. 
Clustering (t-SNE and PCA) found naturally occurring segments of users, affected mostly by the genre of music they listen to. 
It's possible to say with high confidence what songs are listened together through apriori algorithm. 
Interestingly, users, who listen to reggae, are the users who have the most outlying values in majority of dimensions. 
Features, which are most interesting to focus on, are: a) user music taste b) typical driving conditions 
c) co-occuring music genres. 

### Recommendation models

#### Deep recommender with Spotlight framework

##### Rationale

**[CarMusic deep recommender.ipynb](https://bitbucket.org/oskar-j/scalac/src/master/Music_InCarMusic/CarMusic%20deep%20recommender.ipynb)** jupyter notebook 
presents an example model which uses deep learning 
for recommending items (songs) to users, with help of PyTorch and Spotlight.
We check different success metrics, including: *a) RMSE*, *b) Average MRR*, 
*c) precision at k*, *d) recall at k*. 

##### Findings

Despite no user and item features used by the algorithms, we managed to get `1.5` RMSE and `0.2` Precision at K=10. 
For RMSE, the performance is below algorithms like ALS or SVD. Yet, precision was closer to current state of art 
algorithms like SAR or NCF. Reference - [Best Practices on Recommendation Systems by Microsoft](https://github.com/microsoft/recommenders)

#### LightFM collaborative filtering and hybrid models

##### Rationale

**[CarMusic cf and hybrid.ipynb](https://bitbucket.org/oskar-j/scalac/src/master/Music_InCarMusic/CarMusic%20cf%20and%20hybrid.ipynb)** jupyter notebook 
which shows an example hybrid collaborative-filtering model
for recommending items (songs) to users, with help of the LightFM algorithm.

##### Findings

LightFM algorithm was able to recommend songs with mediocre results, with a performance similar to collaborative filtering done by the deep recommender 'spotlight' framework.
We got mean Precision at K: `0.109` and mean Recall at K: `0.135`.

#### Classic machine learning - binary offline recommender 

##### Rationale

**[CarMusic balanced machine learning.ipynb](https://bitbucket.org/oskar-j/scalac/src/master/Music_InCarMusic/CarMusic%20balanced%20machine%20learning.ipynb)** jupyter notebook 
presents some machine learning techniques to make
a binary classifier which predicts whether the user is inclined to listen to a particular song (or not).

##### Findings

We achieved *F1-score* of `0.790 (± 0.013)`. The best configuration, which made for such result, is a 
GradientBoostingClassifier (with input from SimpleImputer), after the hyper-parameter optimization.
Second-best type of algorithm was the HistGradientBoostingClassifier.

#### AutoML approach

##### Rationale

Here, in **[CarMusic AutoML.ipynb](https://bitbucket.org/oskar-j/scalac/src/master/Music_InCarMusic/CarMusic%20AutoML.ipynb)** notebook, 
we show how easy it is to create a classifier with minimal lines of code.
We prepared the learn set by adding negative samples, this tabular input can be later understood by the
TPOT framework for automatic machine learning.

##### Findings

We achieve *F1-score* equal to `0.796`. The pipeline, which was found by the genetic search of the AutoML framework, consists of 
PolynomialFeatures transformer and the GradientBoostingClassifier. This is slightly better compared to manually coded
solution from the ["Classic machine learning - binary offline recommender"](#markdown-header-classic-machine-learning-binary-offline-recommender) section.
