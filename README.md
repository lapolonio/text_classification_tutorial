# Enterprise Text Classification using BERT Tutorial

Step 1: Gather Data

Step 2: Explore BERT Repo to understand steps

Step 3: Modify BERT Repo for needs


### Results on Text Classification from https://github.com/zihangdai/xlnet

Model | IMDB | Yelp-2 | Yelp-5 | DBpedia | Amazon-2 | Amazon-5
--- | --- | --- | --- | --- | --- | ---
BERT-Large | 4.51 | 1.89 | 29.32 | 0.64 | 2.63 | 34.17
XLNet-Large | **3.79** | **1.55** | **27.80** | **0.62** | **2.40** | **32.26**

The above numbers are error rates.


Resources: 
    * https://ai.stanford.edu/~amaas/data/sentiment/
    * https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub
    * https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=SCZWZtKxObjh