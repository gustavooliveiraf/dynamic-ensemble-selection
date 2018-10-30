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

from sklearn.neighbors import KNeighborsClassifier
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class Main:
    def __init__(self, x, y):
        self.data = data
        self.y = np.array(y)
        self.x = np.array(x)
        self.kNeighbors = 3 # lembrar de testar com 3. atual = 5
        self.threshold = 0.5
        self.kdnGreater = [] # ThanThreshold
        self.kdnLess = [] # ThanThreshold
        self.pool_size = 100

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

    def calc_metrics(self, y_samples, y_true):
        proc_auc_score_temp = roc_auc_score(y_samples, y_true)
        geometric_mean_score_temp = geometric_mean_score(y_samples, y_true)
        f1_score_temp = f1_score(y_samples, y_true)

        return (proc_auc_score_temp, geometric_mean_score_temp, f1_score_temp)

# =============================================================================================================================

    def DES(self, x_train, y_train,X_test, Y_test, dsel):
        pool_classifiers = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        pool_classifiers.fit(x_train, y_train)

        # Initialize the DES model
        knorae = KNORAE(pool_classifiers)
        knorau = KNORAU(pool_classifiers)

        # Preprocess the Dynamic Selection dataset (DSEL)
        score1 = knorae.fit(x_train[dsel], y_train[dsel])
        score2 = knorau.fit(x_train[dsel], y_train[dsel])

        # Predict new examples:
        # print (knorae.score(X_test, Y_test), knorau.score(X_test, Y_test))
        return (score1, score2, ) + self.calc_metrics(X_test, Y_test)

                                        # ------------------------------------------ #

    def DCS(self, x_train, y_train,X_test, Y_test, dsel):
        pool_classifiers = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        pool_classifiers.fit(x_train, y_train)

        # Initialize the DES model
        lca = LCA(pool_classifiers)
        ola = OLA(pool_classifiers)

        # Preprocess the Dynamic Selection dataset (DSEL)
        score1 = lca.fit(x_train[dsel], y_train[dsel])
        score2 = ola.fit(x_train[dsel], y_train[dsel])

        # Predict new examples:
        # print (lca.score(X_test, Y_test), ola.score(X_test, Y_test))
        return (score1, score2, ) + self.calc_metrics(X_test, Y_test) # dependendo da base formato nao suportado

# =============================================================================================================================

    def gustavo_method(self, x_train, y_train, X_test, Y_test, dsel):
        neigh = KNeighborsClassifier(n_neighbors=self.kNeighbors)
        neigh.fit(self.x[dsel], self.y[dsel])

        pool_classifiers = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        pool_classifiers.fit(x_train, y_train)

        knorau = KNORAU(pool_classifiers)
        knorau.fit(x_train[dsel], y_train[dsel])

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)

        score = []        
        for i in range(len(Y_test)):
            kdn = sum(self.y[neigh.kneighbors([self.x[dsel]], return_distance=False)[0]]!=self.y[i])/self.kNeighbors
            if (kdn > self.threshold):
                score.append(knorau.predict(self.x[i]))
            else:
                score.append(knn.predict(self.x[i]))
        return (score, ) + self.calc_metrics(X_test, Y_test) # dependendo da base formato nao suportado

# =============================================================================================================================

    def architecture(self, k_fold, n_times):
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                skf2 = StratifiedKFold(n_splits=4, shuffle=True)
                dsel = []
                for aux, dsel_aux in skf2.split(self.x[train_index], self.y[train_index]):
                    train_index = aux
                    dsel = dsel_aux
                    break

                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                # x_train, y_train = X_train, Y_train
                # x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                # adequar base antes de rodar, dependendo da base formato nao suportado
                my_method = self.gustavo_method(x_train, y_train, X_test, Y_test, dsel)
                dcs = self.DCS(x_train, y_train, X_test, Y_test, dsel)
                des = self.DES(x_train, y_train, X_test, Y_test, dsel)

                print(my_method, dcs, des)
                # break

# =============================================================================================================================

def test_pruninge(modelo):
    modelo.architecture(10, 1)

# test
# data = pd.read_csv('./cm1.csv')
# data = pd.read_csv('./jm1.csv')
data = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
data.CLASS = enc.fit_transform(data.CLASS)

y = np.array(data["CLASS"])
x = np.array(data.drop(axis=1, columns = ["CLASS"]))

scaler = StandardScaler()
x = scaler.fit_transform(x)

modelo = Main(x, y)

test_pruninge(modelo)