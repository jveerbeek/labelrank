import numpy as np

from sklearn.preprocessing import normalize
from sklearn.exceptions import NotFittedError

from skmultilearn.cluster import LabelCooccurrenceGraphBuilder


class LabelRank:
    """
    A simple Python implementation of the LabelRank method proposed by
     Bin Fu (2018) in 'Learning label dependency for multi-label classification'
    """

    def __init__(self, a=0.3, tol=0.01):
        self.a = a
        self.tol = tol
        self.T = None
        self.graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)


    def fit(self, y):
        y = np.array(y)
        edge_map = self.graph_builder.transform(y)
        W = np.zeros((y.shape[1], y.shape[1]))
        for target, source in edge_map:
            W[target][source] = edge_map[(target, source)]
            W[source][target] = edge_map[(target, source)]
        S = normalize(W, norm='l1', axis=0) 
        self.T = normalize(S, norm='l1', axis=1)


    def transform(self, probas):
        if self.T is None:
            raise NotFittedError('Model is not fitted. Fit LabelRank model first.')
        probas = np.array(probas)
        transformed_probas = []
        for proba in probas:
            p_x_t = proba
            while True:
                p_x_t_1 = self.a * self.T.dot(p_x_t) + ((1 - self.a) * proba)
                diff = (p_x_t_1 - p_x_t) / p_x_t_1 * 100
                p_x_t = p_x_t_1
                if sum(abs(diff)) < self.tol:
                    transformed_probas.append(p_x_t)
                    break
        return np.array(transformed_probas)
