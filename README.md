Credit Card Fraud Detection using Machine Learning
====

Credit card fraud is a growing issue with many challenges including temporal drift and heavy class imbalance. This project attempts to tackle class imbalance using state-of-the-art techniques including Adaptive Synethtic Sampling Approach (ADASYN) and Synethetic Minority Oversampling Technique (SMOTE). Over 280k real transactions made in Europe in September 2013 [1] are used as the training dataset. Three types of machine learning models are compared: Random Forest, Support Vector Machine, and Multi-Layer Perceptron. Results show that the optimal sampling method for an imbalanced dataset is dependent on the dataset and the model being used.

This project has the following components:

a) IEEE style Paper in PDF format 

b) Jupyter Notebook walking through machine learning tests conducted. You can run view and run them yourself. Included are also comments, reasoning, and figures. For your convenience I have included a copy of the original dataset [1] in this git repo, however please refer to the original source for the most up-to-date version.

This project was done as part of SYDE 522: Machine Learning at the University of Waterloo in Winter 2017.

Installation
-----------

1. Clone the project:

   `$ git clone https://github.com/yazanobeidi/fraud-detection.git && cd fraud-detection`

2. Pip-install dependencies. For example using a `virtualenv`:

   `$ virtualenv env && source env/bin/activate && pip install -r requirements.txt`

Usage
-----
a) Read the Paper (PDF): 

`credit_card_fraud_detection_yazan_obeidi.pdf`

b) Run the Jupyter Notebook:

1. First unzip the dataset:

`$ unzip data/creditcardfraud.zip`

2. Generate a balanced dataset using ADASYN resampling (this will take several minutes):

`$ python adasyn.py`

3. Run the notebook:

`$ jupyter notebook`

Authors
------------
Yazan Obeidi

Copyright
------------
2017, Yazan Obeidi

References
------------
[1] Kaggle. (2017, Jan. 12). Credit Card Fraud Detection [Online]. Available: https://www.kaggle.com/dalpozz/creditcardfraud
