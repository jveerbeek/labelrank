 # LabelRank
 A simple Python implementation of the LabelRank method proposed by Bin Fu (2018) in ['Learning label dependency for multi-label classification'](https://opus.lib.uts.edu.au/bitstream/10453/123252/2/02whole.pdf). Exploits multi-label dependencies  

### Dependencies:
* scikit-learn
* [scikit-multilearn](http://scikit.ml/index.html)

### Example usage:

```python
from labelrank import LabelRank

from skmultilearn.dataset import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import label_ranking_average_precision_score

# Train a regular, binary relevance Logistic Regression classifier, on the emotions dataset.  
X, y, feature_names, label_names = load_dataset('emotions', 'undivided')
X_train, X_test, y_train, y_test  = train_test_split(X, y, random_state=42, test_size=0.2)
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', 
                                             max_iter=10000,
                                            random_state=42))
clf.fit(X_train, y_train)

predictions = clf.predict_proba(X_test)
print(label_ranking_average_precision_score(y_test.toarray(), predictions))
# 0.8084033613445377


lr = LabelRank(a=0.3)
lr.fit(y_train)
predictions_lr = lr.transform(predictions)
print(label_ranking_average_precision_score(y_test.toarray(), predictions_lr))

# 0.8237628384687208

```

