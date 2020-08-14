 # LabelRank
 A simple Python implementation of the LabelRank method proposed by Bin Fu (2018) in ['Learning label dependency for multi-label classification'](https://opus.lib.uts.edu.au/bitstream/10453/123252/2/02whole.pdf). Exploits multi-label dependencies using only the co-occurences of labels in the training set, and can thus be used for any model that produces probabilities.      

### Dependencies:
* numpy
* scikit-learn
* [scikit-multilearn](http://scikit.ml/index.html)

### Example usage:

```python
from labelrank import LabelRank

from skmultilearn.dataset import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Train a regular, binary relevance Logistic Regression classifier on the emotions dataset
X, y, feature_names, label_names = load_dataset('emotions', 'undivided')
X_train, X_test, y_train, y_test  = train_test_split(X, y, random_state=42, test_size=0.2)
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', 
                                             max_iter=10000,
                                            random_state=42))
clf.fit(X_train, y_train)

# Get the average precision (AP) for the binary relevance classifier
predictions = clf.predict_proba(X_test)
print(label_ranking_average_precision_score(y_test.toarray(), predictions))
# 0.8084033613445377

# Fit the LabelRank model and update predictions
lr = LabelRank(a=0.3, tol=0.01)
lr.fit(y_train)
predictions_lr = lr.transform(predictions)
print(label_ranking_average_precision_score(y_test.toarray(), predictions_lr))
# 0.8237628384687208

```

