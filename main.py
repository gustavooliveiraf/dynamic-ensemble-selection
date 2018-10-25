import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
import seaborn as sns
from collections import Counter
from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 

from sklearn import tree
from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class Main:
    def __init__(self, x, y):
        self.data = data
        self.y = np.array(y)
        self.x = np.array(x)
        self.kNeighbors = 5 # lembrar de testar com 7 ============================
        self.threshold = 0.5
        self.kdnGreater = [] # ThanThreshold
        self.kdnLess = [] # ThanThreshold
        self.pool_size = 100

    def kappa_pruning(self, k_fold, n_times, pool_size, m):
        comb = combinations(range(pool_size), 2)
        pruning = []
        pruningKdnGreater = []
        pruningKdnLess = []
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                # x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                kdnGreater, kdnLess = self.k_Disagreeing_neighbors_kDN(x_train, y_train)

                # X_validationGreater, X_validationLess = self.x[kdnGreater], self.x[kdnLess]
                # Y_validationGreater, Y_validationLess = self.y[kdnGreater], self.y[kdnLess]

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), pool_size)
                BagPercep.fit(x_train, y_train)
                for tupla in comb:
                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(x_train), BagPercep.estimators_[tupla[1]].predict(x_train))
                    pruning.append(tupla + (kappa,))

                    # kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationGreater), BagPercep.estimators_[tupla[1]].predict(X_validationGreater))
                    # pruningKdnGreater.append(tupla + (kappa,))

                    # kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationLess), BagPercep.estimators_[tupla[1]].predict(X_validationLess))
                    # pruningKdnLess.append(tupla + (kappa,))
                break
        
        pruning.sort(key=lambda tup: tup[2])

        return (pruning[:m], pruningKdnGreater[:m], pruningKdnLess[:m])

    def k_Disagreeing_neighbors_kDN(self, x_train, y_train):
        neigh = KNeighborsClassifier(n_neighbors=self.kNeighbors)
        neigh.fit(self.x, self.y)

        kdnGreater = []
        kdnLess = []
        for i in range(len(y_train)):
            kdn = sum(self.y[neigh.kneighbors([x_train[i]], return_distance=False)[0]]!=y_train[i])/self.kNeighbors
            if (kdn > self.threshold):
                kdnGreater.append(i)
            else:
                kdnLess.append(i)

        return (kdnGreater, kdnLess)

# =============================================================================================================================

    def sort_score(self, pool, x_test, y_test):
        temp = pool.estimators_[:]

        score_tuple = []
        for i, percep_i in enumerate(temp):
            score_tuple.append((i, percep_i.score(x_test, y_test)))

        score_tuple.sort(key=lambda tup: tup[1], reverse=True)
        score = []
        for i in score_tuple:
            score.append(i[0])

        # print("=============")
        # print("score_tuple", score_tuple[0][1])

        return score

    def calc_metrics(self, y_samples, y_true):
        proc_auc_score_temp = roc_auc_score(y_samples, y_true)
        geometric_mean_score_temp = geometric_mean_score(y_samples, y_true)
        f1_score_temp = f1_score(y_samples, y_true)

        return (proc_auc_score_temp, geometric_mean_score_temp, f1_score_temp)

    def pairwise_diversity_measure(self, BagPercep, ensemble_size, x):
        comb = combinations(range(ensemble_size), 2)
        kappa = 0
        for tupla in comb:
            kappa += cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(x), BagPercep.estimators_[tupla[1]].predict(x))

        return (2/(ensemble_size*(ensemble_size-1)))*kappa

    def disagreement_diversity_measure(self, BagPercep, ensemble_size, x):
        comb = combinations(range(ensemble_size), 2)
        accumulator, size_x = 0, len(x)
        for tupla in comb:
            differs = sum(np.array(BagPercep.estimators_[tupla[0]].predict(x)) != np.array(BagPercep.estimators_[tupla[1]].predict(x)))
            accumulator += differs/size_x

        return (2/(ensemble_size*(ensemble_size-1)))*accumulator



# =============================================================================================================================

    def pruning(self, k_fold, n_times):
        score_pruning = 0
        score_pool = 0

        score_pruning = (0,0,0,0,0,0)
        score_pool = (0,0,0,0,0,0)
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                # x_train, y_train = X_train, Y_train
                # x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                # =========
                kdnGreater, kdnLess = self.k_Disagreeing_neighbors_kDN(x_train, y_train)
                X_validationGreater, X_validationLess = x_train[kdnGreater], x_train[kdnLess]
                Y_validationGreater, Y_validationLess = y_train[kdnGreater], y_train[kdnLess]
                # ==========

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
                BagPercep.fit(x_train, y_train)
                kappa_diversity = self.pairwise_diversity_measure(BagPercep, len(BagPercep.estimators_), X_test)
                disagreement_diversity_ = self.disagreement_diversity_measure(BagPercep, len(BagPercep.estimators_), X_test)
                score_pool_temp = (BagPercep.score(X_test, Y_test), ) + self.calc_metrics(BagPercep.predict(X_test), Y_test) + (kappa_diversity, disagreement_diversity_, )
                
                score_pool = tuple(map(sum, zip(score_pool, score_pool_temp)))

                # score = self.sort_score(BagPercep, X_validationGreater, Y_validationGreater)
                # score_pruning_temp = self.reduce_error(BagPercep, score, x_train, y_train, X_validationGreater, Y_validationGreater, X_test, Y_test)

                # score = self.sort_score(BagPercep, X_validationLess, Y_validationLess)
                # score_pruning_temp = self.reduce_error(BagPercep, score, x_train, y_train, X_validationLess, Y_validationLess, X_test, Y_test)
                
                score = self.sort_score(BagPercep, x_train, y_train) # so executar uma vez ------------
                score_pruning_temp = self.reduce_error(BagPercep, score, x_train, y_train, x_train, y_train, X_test, Y_test)

                # score_pruning_temp = self.best_first(BagPercep, score, x_train, y_train, X_validationGreater, Y_validationGreater, X_test, Y_test)
                # score_pruning_temp = self.best_first(BagPercep, score, x_train, y_train, X_validationLess, Y_validationLess, X_test, Y_test)
                # score_pruning_temp = self.best_first(BagPercep, score, x_train, y_train, x_train, y_train, X_test, Y_test)
                score_pruning = tuple(map(sum, zip(score_pruning, score_pruning_temp)))

                print(score_pool_temp, "---", score_pruning_temp)
                print("=================")

        return (tuple(map(lambda x: x/k_fold, score_pool)), tuple(map(lambda x: x/k_fold, score_pruning)))

    def reduce_error(self, pool, score_index, x_train, y_train, x_validation, y_validation, x_test, y_test):
        BagPercepCurrent = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        BagPercepCurrent.fit(x_train, y_train)

        ensemble_index = set()
        ensemble_index.add(score_index[0])

        ensemble = []
        ensemble.append(pool.estimators_[score_index[0]])

        BagPercepCurrent.estimators_ = ensemble
        best_score = BagPercepCurrent.score(x_validation, y_validation)
        # metrics = (None, None, None, None)
        while (True):
            index_best_score = 0
            BagPercepCurrent.estimators_ = ensemble
            best_score_test = BagPercepCurrent.score(x_test, y_test)

            metrics = (best_score_test,) + self.calc_metrics(BagPercepCurrent.predict(x_test), y_test)

            for i in list(score_index):
                if i not in ensemble_index:
                    BagPercepCurrent.estimators_ = ensemble + [pool.estimators_[i]]
                    score_current = BagPercepCurrent.score(x_validation, y_validation)

                    if best_score < score_current:
                        best_score = score_current
                        index_best_score = i
            if index_best_score != 0:
                ensemble_index.add(index_best_score)
                ensemble.append(pool.estimators_[index_best_score])
            else:
                # print("best index", len(ensemble), best_score, best_score_test)
                kappa_diversity = self.pairwise_diversity_measure(BagPercepCurrent, len(BagPercepCurrent.estimators_), x_test)
                disagreement_diversity_ = self.disagreement_diversity_measure(BagPercepCurrent, len(BagPercepCurrent.estimators_), x_test)
                return (metrics) + (kappa_diversity, disagreement_diversity_, )
            if len(ensemble_index) == self.pool_size:
                return (metrics) + (kappa_diversity, disagreement_diversity_, )

    def best_first(self, pool, score_index, x_train, y_train,  x_validation, y_validation, x_test, y_test):
        BagPercepCurrent = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        BagPercepCurrent.fit(x_train, y_train)

        BagPercepCurrent.estimators_ = [pool.estimators_[score_index[0]]]
        best_score = BagPercepCurrent.score(x_validation, y_validation)
        best_score_test = BagPercepCurrent.score(x_test, y_test)
        metrics = (best_score_test,) + self.calc_metrics(BagPercepCurrent.predict(x_test), y_test)
        best_index = 1
        best_score_test = 0
        diversity_kappa = 0
        for i, j in enumerate(list(score_index[1:])):
            BagPercepCurrent.estimators_ += [pool.estimators_[j]]
            score_current = BagPercepCurrent.score(x_validation, y_validation)

            if best_score < score_current:
                best_score = score_current
                best_index = i
                best_score_test = BagPercepCurrent.score(x_test, y_test)
                metrics = (best_score_test,) + self.calc_metrics(BagPercepCurrent.predict(x_test), y_test)
                diversity_kappa = self.pairwise_diversity_measure(BagPercepCurrent, len(BagPercepCurrent.estimators_), x_teste)

        best_index += 2
        # print("best index", best_index, best_score, best_score_test)
        return (metrics) +  (diversity_kappa,)

# =============================================================================================================================

    # def 
def test_kappa(modelo):
    # modelo.kNeighborsClassifier()
    best_classifiers = modelo.kappa_pruning(10, 1, 10, 5) #retorna os 3 conjuntos de classificadores a b c
    ensemble = set()
    ensembleGreater = set()
    ensembleLess = set()
    for i in best_classifiers[0]:
        ensemble.add(i[0])
        ensemble.add(i[1])

    for i in best_classifiers[1]:
        ensembleGreater.add(i[0])
        ensembleGreater.add(i[1])

    for i in best_classifiers[2]:
        ensembleLess.add(i[0])
        ensembleLess.add(i[1])

    print(list(ensemble))

def test_pruninge(modelo):
    print(modelo.pruning(10, 1))

# test
data = pd.read_csv('./cm1.csv')
# data = pd.read_csv('./jm1.csv')
# data = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
data.CLASS = enc.fit_transform(data.CLASS)

y = np.array(data["CLASS"])
x = np.array(data.drop(axis=1, columns = ["CLASS"]))

scaler = StandardScaler()
x = scaler.fit_transform(x)

modelo = Main(x, y)

test_pruninge(modelo)