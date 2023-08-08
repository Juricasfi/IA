# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:19:53 2022

@author: juric
"""

import pickle                                # Guardar el modelo clasificador
import xgboost as xgb
import seaborn as sns; sns.set()
import numpy as np

#import matplotlib as plt
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns




from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from numpy import savetxt

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score



#ElM import



from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler

#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from numpy import savetxt





import random

print('Random number with seed 30')
for i in range(3):
    # Random number with seed 30
    random.seed(30)
    print(random.randint(25, 50))


random.seed(30)

rng = np.random.RandomState(31337)
np.random.seed(0)






YTODO = pd.read_csv("Y.csv")
XTODO = pd.read_csv("X.csv")



#YTODO= pd.read_csv("Y.csv")
#XTODO= pd.read_csv("X.csv")


XTODO = pd.read_csv("X.csv")
X = XTODO
X = X.to_numpy()


print("Iris: multiclass classification")
#iris = load_iris()
#y = iris['target']
#X = iris['data']

y = YTODO
X = XTODO
y = y.to_numpy()
y111 = y
X = X.to_numpy()
y = np.ravel(y)








linf=0;
lsup=1;
n=30; #para buscar los vecinos original 50,  $$19 para target dimensions, %%9 para mindist,%% 20 para nh
deltax=(lsup-linf)/(n-1);
u=np.zeros(n);
vectacc=np.zeros(n);
vectacc2=np.zeros(n);
x=np.linspace(2,32,n,endpoint=False); #para buscar los vecinos original 6,56  %%2,21 para target dimensions %%0.1,1 para mindist
#x=np.linspace(1,17,n,endpoint=False); #para buscar las dimensiones
#x=np.linspace(10,210,n,endpoint=False); #para buscar nh hiiden nodes 10,210
accuracy_vecinos=pd.DataFrame()
accuracy_vecinos_2 = pd.DataFrame()
for i in range(0,30) :
    #u[i+1]=(u[i+1])*(deltax+1)+(deltax*(1-x[i+1]))  
    vectx= x[i]
    np.random.seed(0)
    
    
    n_neighbors = vectx.astype(int)
    #n_neighbors = 16
    
    
    #min_dist = vectx.astype(float)
    min_dist = 0.6  

                           #UMAP
    #n_components = vectx.astype(int)
    n_components = 8 

                          #dimension de destino
    
    XTODO = pd.read_csv("X.csv")
    X = XTODO
    X = X.to_numpy()
    YTODO = pd.read_csv("Y.csv")
    y = YTODO
    y = y.to_numpy()
    y = np.ravel(y)
    
  
        
        
    ######### metric learning supervised umap inicio##########################    
    
    X_train, X_test, y_train, y_test = train_test_split(X,      ##divide los datos en 75% para train y 25% para test por defecto
                                                        y,
                                                        stratify=y,
                                                        random_state=42)
    
    
    
    
    mapper = umap.UMAP(n_neighbors=n_neighbors,  #100 para 6gases 
    min_dist=min_dist,   #0.5 para 6gases #
    n_components=n_components,
    random_state=42,).fit(X_train, y_train)
   
    
    #sacar el embedding del train
    train_embedding = mapper.transform(X_train)
    savetxt('supXtrainumap600X8.csv', train_embedding, delimiter=',')
    

    ################sacar el embedding del test
    test_embedding = mapper.transform(X_test)
    savetxt('supXtestumap600X8.csv', test_embedding, delimiter=',')
    

####Nuevos X e y despues de reducción de dimensionalidad

#train
    X = train_embedding
    y = y_train   
    
#test    
    
    X_X_test = test_embedding
    y_y_test = y_test
    
    
    ######5 fold cross validation Clasificador ELM   inicio#########
    
    def make_classifiers():
        """
    
        :return:
        """
        #names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        names = ["ELM(10,tanh)",  "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        nh = 10
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        rng = np.random.RandomState(31337) 
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
    
        classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                       #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                       GenELMClassifier(hidden_layer=srhl_sinsq),
                       GenELMClassifier(hidden_layer=srhl_tribas),
                       GenELMClassifier(hidden_layer=srhl_hardlim),
                       GenELMClassifier(hidden_layer=srhl_rbf)]
    
        return names, classifiers
    
        
        
    if __name__ == '__main__':
        # generate some datasets
        #datasets = make_datasets()
        names, classifiers = make_classifiers()
    
        
        
    
    
        ############kfold
    rng = np.random.RandomState(31337)    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    
    s = (6,6)
    conf=np.zeros(s)
    
    
    
    for train_index, test_index in kf.split(X):
        
        nh =60       #número de nodos ocultos 
        #nh= vectx.astype(int)
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
        
        
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        #clf = GenELMClassifier(hidden_layer=srhl_sinsq)
        #clf = GenELMClassifier(hidden_layer=srhl_tribas)
        #clf = GenELMClassifier(hidden_layer=srhl_hardlim)
        #clf = GenELMClassifier(hidden_layer=srhl_rbf)
        
        neigh_model = clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        actuals = y[test_index]
        conf1=confusion_matrix(actuals, predictions)
        conf += conf1
        
        print(confusion_matrix(actuals, predictions))
    
    ax = sns.heatmap(conf,cmap="Wistia", annot=True, fmt='.0f')
    

    savetxt('data.csv', conf, delimiter=',')
    
    
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    
    print(accuracy(conf))
    accu1 = accuracy(conf)
    
    
    
    
     ### nehigbors =16
    
    
    
    
     ######5 fold cross validation Clasificador ELM   final#########
     
     
    conf1=conf   
    
    def accuraciesNPM(conf1):
        accuracyNPM = 0;
        precisionNPM = 0;
        recallNPM = 0;
        specificityNPM = 0;
        F1scoreNPM = 0;
        for i in [0,1,2,3,4,5]:
          tpi=conf1[i,i]      
          fpi=0
          fni=0
          tni=0
          for k in [0,1,2,3,4,5]:
            if k != i:
              fpi += conf1[k,i]
              fni += conf1[i,k]
              for j in [0,1,2,3,4,5]:
                if j != i:
                  tni += conf1[k,j]
          accuracyNPM += (tpi+tni)/(tpi+tni+fni+fpi)/6 
          precisionNPM += tpi/(tpi+fpi)/6
          recallNPM += tpi/(tpi+fni)/6
          specificityNPM += tni/(tni+fpi)/6
        F1scoreNPM =2*precisionNPM*recallNPM/(precisionNPM + recallNPM)
        return accuracyNPM , precisionNPM , specificityNPM , recallNPM , F1scoreNPM
    
    confusionmia = accuraciesNPM(conf1)
    print(accuraciesNPM(conf1))   
    
    
    accu11 = accuraciesNPM(conf1)
    
    accu_veci = [*accu11]
    accu_veci.append(accu1)
    
    accu_veci = pd.DataFrame(accu_veci)
    accu_veci = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    # vectacc[i] = confusionmia[0]
    # vectacc[i+1]= vectacc[i]


    
    
    pickl = {'model': clf}
    pickle.dump(pickl, open('model_file'+'.p','wb'))
    
    
    file_name = 'model_file.p'
    with open(file_name,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    #input transformed values to make a prediction in FlaskAPI
    model.predict(X[4,:].reshape(1,-1))
    model.predict(X[60,:].reshape(1,-1))
    
    model.predict(X_X_test[60,:].reshape(1,-1))
    
    
    predictionsegundoyear= model.predict(X_X_test)
    
    confsegundoyear=confusion_matrix(y_y_test, predictionsegundoyear)
    savetxt('datatest.csv', confsegundoyear, delimiter=',')
    #print(accuracy(confsegundoyear))
    confusionmia2 = accuraciesNPM(confsegundoyear)
    
    print(accuraciesNPM(confsegundoyear))
    
    accu2=accuracy(confsegundoyear)
    accu22 = accuraciesNPM(confsegundoyear)
    
    accu_veci_2 = [*accu22]
    accu_veci_2.append(accu2)
    
    accu_veci_2 = pd.DataFrame(accu_veci_2)
    accu_veci_2 = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    accuracy_vecinos[x[i]] = accu_veci
    accuracy_vecinos_2[x[i]] = accu_veci_2
    
    vectacc2[i] = confusionmia2[0]
    # vectacc2[i+1]= vectacc2[i]
    
    
    
    
    del confusionmia
    del confusionmia2
    del X
    del y

accuracy_vecinos = accuracy_vecinos.transpose()


plt.plot(x,accuracy_vecinos['Accuracy'])



####################################################################################################
###############################################################################################
#####################  Mínima Distancia #################################################


y = YTODO
X = XTODO
y = y.to_numpy()
y111 = y
X = X.to_numpy()
y = np.ravel(y)






del accu_veci
del accu_veci_2

linf=0;
lsup=1;
n=19; #para buscar los vecinos original 50,  $$19 para target dimensions, %%9 para mindist,%% 20 para nh
deltax=(lsup-linf)/(n-1);
u=np.zeros(n);
vectacc=np.zeros(n);
vectacc2=np.zeros(n);
x=np.linspace(0.05,1,n,endpoint=False); #para buscar los vecinos original 6,56  %%2,21 para target dimensions %%0.1,1 para mindist
#x=np.linspace(1,17,n,endpoint=False); #para buscar las dimensiones
#x=np.linspace(10,210,n,endpoint=False); #para buscar nh hiiden nodes 10,210
accuracy_min_dist=pd.DataFrame()
accuracy_min_dist_2 = pd.DataFrame()
for i in range(0,19) :
    #u[i+1]=(u[i+1])*(deltax+1)+(deltax*(1-x[i+1]))  
    vectx= x[i]
    np.random.seed(0)
    
    
    #n_neighbors = vectx.astype(int)
    n_neighbors = 16
    
    
    min_dist = vectx.astype(float)
    #min_dist = 0.6  

                           #UMAP
    #n_components = vectx.astype(int)
    n_components = 8 

                          #dimension de destino
    
    XTODO = pd.read_csv("X.csv")
    X = XTODO
    X = X.to_numpy()
    YTODO = pd.read_csv("Y.csv")
    y = YTODO
    y = y.to_numpy()
    y = np.ravel(y)
    
  
        
        
    ######### metric learning supervised umap inicio##########################    
    
    X_train, X_test, y_train, y_test = train_test_split(X,      ##divide los datos en 75% para train y 25% para test por defecto
                                                        y,
                                                        stratify=y,
                                                        random_state=42)
    
    
    
    
    mapper = umap.UMAP(n_neighbors=n_neighbors,  #100 para 6gases 
    min_dist=min_dist,   #0.5 para 6gases #
    n_components=n_components,
    random_state=42,).fit(X_train, y_train)
   
    
    #sacar el embedding del train
    train_embedding = mapper.transform(X_train)
    savetxt('supXtrainumap600X8.csv', train_embedding, delimiter=',')
    

    ################sacar el embedding del test
    test_embedding = mapper.transform(X_test)
    savetxt('supXtestumap600X8.csv', test_embedding, delimiter=',')
    

####Nuevos X e y despues de reducción de dimensionalidad

#train
    X = train_embedding
    y = y_train   
    
#test    
    
    X_X_test = test_embedding
    y_y_test = y_test
    
    
    ######5 fold cross validation Clasificador ELM   inicio#########
    
    def make_classifiers():
        """
    
        :return:
        """
        #names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        names = ["ELM(10,tanh)",  "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        nh = 10
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        rng = np.random.RandomState(31337) 
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
    
        classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                       #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                       GenELMClassifier(hidden_layer=srhl_sinsq),
                       GenELMClassifier(hidden_layer=srhl_tribas),
                       GenELMClassifier(hidden_layer=srhl_hardlim),
                       GenELMClassifier(hidden_layer=srhl_rbf)]
    
        return names, classifiers
    
        
        
    if __name__ == '__main__':
        # generate some datasets
        #datasets = make_datasets()
        names, classifiers = make_classifiers()
    
        
        
    
    
        ############kfold
    rng = np.random.RandomState(31337)    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    
    s = (6,6)
    conf=np.zeros(s)
    
    
    
    for train_index, test_index in kf.split(X):
        
        nh =60       #número de nodos ocultos 
        #nh= vectx.astype(int)
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
        
        
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        #clf = GenELMClassifier(hidden_layer=srhl_sinsq)
        #clf = GenELMClassifier(hidden_layer=srhl_tribas)
        #clf = GenELMClassifier(hidden_layer=srhl_hardlim)
        #clf = GenELMClassifier(hidden_layer=srhl_rbf)
        
        neigh_model = clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        actuals = y[test_index]
        conf1=confusion_matrix(actuals, predictions)
        conf += conf1
        
        print(confusion_matrix(actuals, predictions))
    
    ax = sns.heatmap(conf,cmap="Wistia", annot=True, fmt='.0f')
    

    savetxt('data.csv', conf, delimiter=',')
    
    
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    
    print(accuracy(conf))
    accu1 = accuracy(conf)
    
    
    
    
     ### nehigbors =16
    
    
    
    
     ######5 fold cross validation Clasificador ELM   final#########
     
     
    conf1=conf   
    
    def accuraciesNPM(conf1):
        accuracyNPM = 0;
        precisionNPM = 0;
        recallNPM = 0;
        specificityNPM = 0;
        F1scoreNPM = 0;
        for i in [0,1,2,3,4,5]:
          tpi=conf1[i,i]      
          fpi=0
          fni=0
          tni=0
          for k in [0,1,2,3,4,5]:
            if k != i:
              fpi += conf1[k,i]
              fni += conf1[i,k]
              for j in [0,1,2,3,4,5]:
                if j != i:
                  tni += conf1[k,j]
          accuracyNPM += (tpi+tni)/(tpi+tni+fni+fpi)/6 
          precisionNPM += tpi/(tpi+fpi)/6
          recallNPM += tpi/(tpi+fni)/6
          specificityNPM += tni/(tni+fpi)/6
        F1scoreNPM =2*precisionNPM*recallNPM/(precisionNPM + recallNPM)
        return accuracyNPM , precisionNPM , specificityNPM , recallNPM , F1scoreNPM
    
    confusionmia = accuraciesNPM(conf1)
    print(accuraciesNPM(conf1))   
    
    
    accu11 = accuraciesNPM(conf1)
    
    accu_veci = [*accu11]
    accu_veci.append(accu1)
    
    accu_veci = pd.DataFrame(accu_veci)
    accu_veci = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    # vectacc[i] = confusionmia[0]
    # vectacc[i+1]= vectacc[i]


    
    
    pickl = {'model': clf}
    pickle.dump(pickl, open('model_file'+'.p','wb'))
    
    
    file_name = 'model_file.p'
    with open(file_name,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    #input transformed values to make a prediction in FlaskAPI
    model.predict(X[4,:].reshape(1,-1))
    model.predict(X[60,:].reshape(1,-1))
    
    model.predict(X_X_test[60,:].reshape(1,-1))
    
    
    predictionsegundoyear= model.predict(X_X_test)
    
    confsegundoyear=confusion_matrix(y_y_test, predictionsegundoyear)
    savetxt('datatest.csv', confsegundoyear, delimiter=',')
    #print(accuracy(confsegundoyear))
    confusionmia2 = accuraciesNPM(confsegundoyear)
    
    print(accuraciesNPM(confsegundoyear))
    
    accu2=accuracy(confsegundoyear)
    accu22 = accuraciesNPM(confsegundoyear)
    
    accu_veci_2 = [*accu22]
    accu_veci_2.append(accu2)
    
    accu_veci_2 = pd.DataFrame(accu_veci_2)
    accu_veci_2 = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    accuracy_min_dist[x[i]] = accu_veci
    accuracy_min_dist_2[x[i]] = accu_veci_2
    
    vectacc2[i] = confusionmia2[0]
    # vectacc2[i+1]= vectacc2[i]
    
    
    
    
    del confusionmia
    del confusionmia2
    del X
    del y

accuracy_min_dist = accuracy_min_dist.transpose()


plt.plot(x,accuracy_min_dist['Accuracy'])





##########################################################################################
######################################################################################
#################################  n-componentes #####################################


del accu_veci
del accu_veci_2

linf=0;
lsup=1;
n=19; #para buscar los vecinos original 50,  $$19 para target dimensions, %%9 para mindist,%% 20 para nh
deltax=(lsup-linf)/(n-1);
u=np.zeros(n);
vectacc=np.zeros(n);
vectacc2=np.zeros(n);
x=np.linspace(2,21,n,endpoint=False); #para buscar los vecinos original 6,56  %%2,21 para target dimensions %%0.1,1 para mindist
#x=np.linspace(1,17,n,endpoint=False); #para buscar las dimensiones
#x=np.linspace(10,210,n,endpoint=False); #para buscar nh hiiden nodes 10,210
accuracy_n_comp=pd.DataFrame()
accuracy_n_comp_2 = pd.DataFrame()
for i in range(0,19) :
    #u[i+1]=(u[i+1])*(deltax+1)+(deltax*(1-x[i+1]))  
    vectx= x[i]
    np.random.seed(0)
    
    
    #n_neighbors = vectx.astype(int)
    n_neighbors = 16
    
    
    #min_dist = vectx.astype(float)
    min_dist = 0.6  

                           #UMAP
    n_components = vectx.astype(int)
    #n_components = 8 

                          #dimension de destino
    
    XTODO = pd.read_csv("X.csv")
    X = XTODO
    X = X.to_numpy()
    YTODO = pd.read_csv("Y.csv")
    y = YTODO
    y = y.to_numpy()
    y = np.ravel(y)
    
  
        
        
    ######### metric learning supervised umap inicio##########################    
    
    X_train, X_test, y_train, y_test = train_test_split(X,      ##divide los datos en 75% para train y 25% para test por defecto
                                                        y,
                                                        stratify=y,
                                                        random_state=42)
    
    
    
    
    mapper = umap.UMAP(n_neighbors=n_neighbors,  #100 para 6gases 
    min_dist=min_dist,   #0.5 para 6gases #
    n_components=n_components,
    random_state=42,).fit(X_train, y_train)
   
    
    #sacar el embedding del train
    train_embedding = mapper.transform(X_train)
    savetxt('supXtrainumap600X8.csv', train_embedding, delimiter=',')
    

    ################sacar el embedding del test
    test_embedding = mapper.transform(X_test)
    savetxt('supXtestumap600X8.csv', test_embedding, delimiter=',')
    

####Nuevos X e y despues de reducción de dimensionalidad

#train
    X = train_embedding
    y = y_train   
    
#test    
    
    X_X_test = test_embedding
    y_y_test = y_test
    
    
    ######5 fold cross validation Clasificador ELM   inicio#########
    
    def make_classifiers():
        """
    
        :return:
        """
        #names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        names = ["ELM(10,tanh)",  "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        nh = 10
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        rng = np.random.RandomState(31337) 
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
    
        classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                       #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                       GenELMClassifier(hidden_layer=srhl_sinsq),
                       GenELMClassifier(hidden_layer=srhl_tribas),
                       GenELMClassifier(hidden_layer=srhl_hardlim),
                       GenELMClassifier(hidden_layer=srhl_rbf)]
    
        return names, classifiers
    
        
        
    if __name__ == '__main__':
        # generate some datasets
        #datasets = make_datasets()
        names, classifiers = make_classifiers()
    
        
        
    
    
        ############kfold
    rng = np.random.RandomState(31337)    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    
    s = (6,6)
    conf=np.zeros(s)
    
    
    
    for train_index, test_index in kf.split(X):
        
        nh =60       #número de nodos ocultos 
        #nh= vectx.astype(int)
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
        
        
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        #clf = GenELMClassifier(hidden_layer=srhl_sinsq)
        #clf = GenELMClassifier(hidden_layer=srhl_tribas)
        #clf = GenELMClassifier(hidden_layer=srhl_hardlim)
        #clf = GenELMClassifier(hidden_layer=srhl_rbf)
        
        neigh_model = clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        actuals = y[test_index]
        conf1=confusion_matrix(actuals, predictions)
        conf += conf1
        
        print(confusion_matrix(actuals, predictions))
    
    ax = sns.heatmap(conf,cmap="Wistia", annot=True, fmt='.0f')
    

    savetxt('data.csv', conf, delimiter=',')
    
    
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    
    print(accuracy(conf))
    accu1 = accuracy(conf)
    
    
    
    
     ### nehigbors =16
    
    
    
    
     ######5 fold cross validation Clasificador ELM   final#########
     
     
    conf1=conf   
    
    def accuraciesNPM(conf1):
        accuracyNPM = 0;
        precisionNPM = 0;
        recallNPM = 0;
        specificityNPM = 0;
        F1scoreNPM = 0;
        for i in [0,1,2,3,4,5]:
          tpi=conf1[i,i]      
          fpi=0
          fni=0
          tni=0
          for k in [0,1,2,3,4,5]:
            if k != i:
              fpi += conf1[k,i]
              fni += conf1[i,k]
              for j in [0,1,2,3,4,5]:
                if j != i:
                  tni += conf1[k,j]
          accuracyNPM += (tpi+tni)/(tpi+tni+fni+fpi)/6 
          precisionNPM += tpi/(tpi+fpi)/6
          recallNPM += tpi/(tpi+fni)/6
          specificityNPM += tni/(tni+fpi)/6
        F1scoreNPM =2*precisionNPM*recallNPM/(precisionNPM + recallNPM)
        return accuracyNPM , precisionNPM , specificityNPM , recallNPM , F1scoreNPM
    
    confusionmia = accuraciesNPM(conf1)
    print(accuraciesNPM(conf1))   
    
    
    accu11 = accuraciesNPM(conf1)
    
    accu_veci = [*accu11]
    accu_veci.append(accu1)
    
    accu_veci = pd.DataFrame(accu_veci)
    accu_veci = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    # vectacc[i] = confusionmia[0]
    # vectacc[i+1]= vectacc[i]


    
    
    pickl = {'model': clf}
    pickle.dump(pickl, open('model_file'+'.p','wb'))
    
    
    file_name = 'model_file.p'
    with open(file_name,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    #input transformed values to make a prediction in FlaskAPI
    model.predict(X[4,:].reshape(1,-1))
    model.predict(X[60,:].reshape(1,-1))
    
    model.predict(X_X_test[60,:].reshape(1,-1))
    
    
    predictionsegundoyear= model.predict(X_X_test)
    
    confsegundoyear=confusion_matrix(y_y_test, predictionsegundoyear)
    savetxt('datatest.csv', confsegundoyear, delimiter=',')
    #print(accuracy(confsegundoyear))
    confusionmia2 = accuraciesNPM(confsegundoyear)
    
    print(accuraciesNPM(confsegundoyear))
    
    accu2=accuracy(confsegundoyear)
    accu22 = accuraciesNPM(confsegundoyear)
    
    accu_veci_2 = [*accu22]
    accu_veci_2.append(accu2)
    
    accu_veci_2 = pd.DataFrame(accu_veci_2)
    accu_veci_2 = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    accuracy_n_comp[x[i]] = accu_veci
    accuracy_n_comp_2[x[i]] = accu_veci_2
    
    vectacc2[i] = confusionmia2[0]
    # vectacc2[i+1]= vectacc2[i]
    
    
    
    
    del confusionmia
    del confusionmia2
    del X
    del y

accuracy_n_comp = accuracy_n_comp.transpose()


plt.plot(x,accuracy_n_comp['Accuracy'])







######################################################################################
#######################################################################################
##########################       nh         ####################################
del accu_veci
del accu_veci_2

linf=0;
lsup=1;
n=20; #para buscar los vecinos original 50,  $$19 para target dimensions, %%9 para mindist,%% 20 para nh
deltax=(lsup-linf)/(n-1);
u=np.zeros(n);
vectacc=np.zeros(n);
vectacc2=np.zeros(n);
x=np.linspace(10,210,n,endpoint=False); #para buscar los vecinos original 6,56  %%2,21 para target dimensions %%0.1,1 para mindist
#x=np.linspace(1,17,n,endpoint=False); #para buscar las dimensiones
#x=np.linspace(10,210,n,endpoint=False); #para buscar nh hiiden nodes 10,210
accuracy_nh=pd.DataFrame()
accuracy_nh_2 = pd.DataFrame()
for i in range(0,20) :
    #u[i+1]=(u[i+1])*(deltax+1)+(deltax*(1-x[i+1]))  
    vectx= x[i]
    np.random.seed(0)
    
    
    #n_neighbors = vectx.astype(int)
    n_neighbors = 16
    
    
    #min_dist = vectx.astype(float)
    min_dist = 0.6  

                           #UMAP
    #n_components = vectx.astype(int)
    n_components = 8 

                          #dimension de destino
    
    XTODO = pd.read_csv("X.csv")
    X = XTODO
    X = X.to_numpy()
    YTODO = pd.read_csv("Y.csv")
    y = YTODO
    y = y.to_numpy()
    y = np.ravel(y)
    
  
        
        
    ######### metric learning supervised umap inicio##########################    
    
    X_train, X_test, y_train, y_test = train_test_split(X,      ##divide los datos en 75% para train y 25% para test por defecto
                                                        y,
                                                        stratify=y,
                                                        random_state=42)
    
    
    
    
    mapper = umap.UMAP(n_neighbors=n_neighbors,  #100 para 6gases 
    min_dist=min_dist,   #0.5 para 6gases #
    n_components=n_components,
    random_state=42,).fit(X_train, y_train)
   
    
    #sacar el embedding del train
    train_embedding = mapper.transform(X_train)
    savetxt('supXtrainumap600X8.csv', train_embedding, delimiter=',')
    

    ################sacar el embedding del test
    test_embedding = mapper.transform(X_test)
    savetxt('supXtestumap600X8.csv', test_embedding, delimiter=',')
    

####Nuevos X e y despues de reducción de dimensionalidad

#train
    X = train_embedding
    y = y_train   
    
#test    
    
    X_X_test = test_embedding
    y_y_test = y_test
    
    
    ######5 fold cross validation Clasificador ELM   inicio#########
    
    def make_classifiers():
        """
    
        :return:
        """
        #names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        names = ["ELM(10,tanh)",  "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        nh = 10
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        rng = np.random.RandomState(31337) 
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
    
        classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                       #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                       GenELMClassifier(hidden_layer=srhl_sinsq),
                       GenELMClassifier(hidden_layer=srhl_tribas),
                       GenELMClassifier(hidden_layer=srhl_hardlim),
                       GenELMClassifier(hidden_layer=srhl_rbf)]
    
        return names, classifiers
    
        
        
    if __name__ == '__main__':
        # generate some datasets
        #datasets = make_datasets()
        names, classifiers = make_classifiers()
    
        
        
    
    
        ############kfold
    rng = np.random.RandomState(31337)    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    
    s = (6,6)
    conf=np.zeros(s)
    
    
    
    for train_index, test_index in kf.split(X):
        
        nh = x[i]       #número de nodos ocultos 
        nh= vectx.astype(int)
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
        
        
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        #clf = GenELMClassifier(hidden_layer=srhl_sinsq)
        #clf = GenELMClassifier(hidden_layer=srhl_tribas)
        #clf = GenELMClassifier(hidden_layer=srhl_hardlim)
        #clf = GenELMClassifier(hidden_layer=srhl_rbf)
        
        neigh_model = clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        actuals = y[test_index]
        conf1=confusion_matrix(actuals, predictions)
        conf += conf1
        
        print(confusion_matrix(actuals, predictions))
    
    ax = sns.heatmap(conf,cmap="Wistia", annot=True, fmt='.0f')
    

    savetxt('data.csv', conf, delimiter=',')
    
    
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    
    print(accuracy(conf))
    accu1 = accuracy(conf)
    
    
    
    
     ### nehigbors =16
    
    
    
    
     ######5 fold cross validation Clasificador ELM   final#########
     
     
    conf1=conf   
    
    def accuraciesNPM(conf1):
        accuracyNPM = 0;
        precisionNPM = 0;
        recallNPM = 0;
        specificityNPM = 0;
        F1scoreNPM = 0;
        for i in [0,1,2,3,4,5]:
          tpi=conf1[i,i]      
          fpi=0
          fni=0
          tni=0
          for k in [0,1,2,3,4,5]:
            if k != i:
              fpi += conf1[k,i]
              fni += conf1[i,k]
              for j in [0,1,2,3,4,5]:
                if j != i:
                  tni += conf1[k,j]
          accuracyNPM += (tpi+tni)/(tpi+tni+fni+fpi)/6 
          precisionNPM += tpi/(tpi+fpi)/6
          recallNPM += tpi/(tpi+fni)/6
          specificityNPM += tni/(tni+fpi)/6
        F1scoreNPM =2*precisionNPM*recallNPM/(precisionNPM + recallNPM)
        return accuracyNPM , precisionNPM , specificityNPM , recallNPM , F1scoreNPM
    
    confusionmia = accuraciesNPM(conf1)
    print(accuraciesNPM(conf1))   
    
    
    accu11 = accuraciesNPM(conf1)
    
    accu_veci = [*accu11]
    accu_veci.append(accu1)
    
    accu_veci = pd.DataFrame(accu_veci)
    accu_veci = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    # vectacc[i] = confusionmia[0]
    # vectacc[i+1]= vectacc[i]


    
    
    pickl = {'model': clf}
    pickle.dump(pickl, open('model_file'+'.p','wb'))
    
    
    file_name = 'model_file.p'
    with open(file_name,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    #input transformed values to make a prediction in FlaskAPI
    model.predict(X[4,:].reshape(1,-1))
    model.predict(X[60,:].reshape(1,-1))
    
    model.predict(X_X_test[60,:].reshape(1,-1))
    
    
    predictionsegundoyear= model.predict(X_X_test)
    
    confsegundoyear=confusion_matrix(y_y_test, predictionsegundoyear)
    savetxt('datatest.csv', confsegundoyear, delimiter=',')
    #print(accuracy(confsegundoyear))
    confusionmia2 = accuraciesNPM(confsegundoyear)
    
    print(accuraciesNPM(confsegundoyear))
    
    accu2=accuracy(confsegundoyear)
    accu22 = accuraciesNPM(confsegundoyear)
    
    accu_veci_2 = [*accu22]
    accu_veci_2.append(accu2)
    
    accu_veci_2 = pd.DataFrame(accu_veci_2)
    accu_veci_2 = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    accuracy_nh[x[i]] = accu_veci
    accuracy_nh_2[x[i]] = accu_veci_2
    
    vectacc2[i] = confusionmia2[0]
    # vectacc2[i+1]= vectacc2[i]
    
    
    
    
    del confusionmia
    del confusionmia2
    del X
    del y

accuracy_nh = accuracy_nh.transpose()


plt.plot(x,accuracy_nh['Accuracy'])




## %% #################################################################################
###################################################################################
######################## Función de activación ######################################3
###################################################################################


del accu_veci
del accu_veci_2

linf=0;
lsup=1;
n=4; #para buscar los vecinos original 50,  $$19 para target dimensions, %%9 para mindist,%% 20 para nh
deltax=(lsup-linf)/(n-1);
u=np.zeros(n);
vectacc=np.zeros(n);
vectacc2=np.zeros(n);

accuracy_funcion_activacion=pd.DataFrame()
accuracy_funcion_activacion_2 = pd.DataFrame()
for i in range(0,4) :
    #u[i+1]=(u[i+1])*(deltax+1)+(deltax*(1-x[i+1]))  
    vectx= x[i]
    np.random.seed(0)
    
    
    #n_neighbors = vectx.astype(int)
    n_neighbors = 16
    
    
    #min_dist = vectx.astype(float)
    min_dist = 0.6  

                           #UMAP
    #n_components = vectx.astype(int)
    n_components = 8 

                          #dimension de destino
    
    XTODO = pd.read_csv("X.csv")
    X = XTODO
    X = X.to_numpy()
    YTODO = pd.read_csv("Y.csv")
    y = YTODO
    y = y.to_numpy()
    y = np.ravel(y)
    
  
        
        
    ######### metric learning supervised umap inicio##########################    
    
    X_train, X_test, y_train, y_test = train_test_split(X,      ##divide los datos en 75% para train y 25% para test por defecto
                                                        y,
                                                        stratify=y,
                                                        random_state=42)
    
    
    
    
    mapper = umap.UMAP(n_neighbors=n_neighbors,  #100 para 6gases 
    min_dist=min_dist,   #0.5 para 6gases #
    n_components=n_components,
    random_state=42,).fit(X_train, y_train)
   
    
    #sacar el embedding del train
    train_embedding = mapper.transform(X_train)
    savetxt('supXtrainumap600X8.csv', train_embedding, delimiter=',')
    

    ################sacar el embedding del test
    test_embedding = mapper.transform(X_test)
    savetxt('supXtestumap600X8.csv', test_embedding, delimiter=',')
    

####Nuevos X e y despues de reducción de dimensionalidad

#train
    X = train_embedding
    y = y_train   
    
#test    
    
    X_X_test = test_embedding
    y_y_test = y_test
    
    
    ######5 fold cross validation Clasificador ELM   inicio#########
    
    def make_classifiers():
        """
    
        :return:
        """
        #names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        names = ["ELM(10,tanh)",  "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]
    
        nh = 10
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        rng = np.random.RandomState(31337) 
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()

        classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                       #GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                       GenELMClassifier(hidden_layer=srhl_sinsq),
                       GenELMClassifier(hidden_layer=srhl_tribas),
                       GenELMClassifier(hidden_layer=srhl_hardlim),
                       GenELMClassifier(hidden_layer=srhl_rbf)]
    
        return names, classifiers
    
        
        
    if __name__ == '__main__':
        # generate some datasets
        #datasets = make_datasets()
        names, classifiers = make_classifiers()
    
        
        
    
    
        ############kfold
    rng = np.random.RandomState(31337)    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    
    s = (6,6)
    conf=np.zeros(s)
    
    
    
    for train_index, test_index in kf.split(X):
        
        nh = 60      #número de nodos ocultos 
        #nh= vectx.astype(int)
    
        # pass user defined transfer func
        sinsq = (lambda x: np.power(np.sin(x), 2.0))
        srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)
    
        # use internal transfer funcs
        srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
        srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
        srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')
    
        # use gaussian RBF
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=rng)
        log_reg = LogisticRegression()
        
        if i == 0:
            clf = GenELMClassifier(hidden_layer=srhl_tanh)
        else:
            if i == 1:
                clf = GenELMClassifier(hidden_layer=srhl_tribas)
                    
            else:
                if i == 2:
                    clf = GenELMClassifier(hidden_layer=srhl_hardlim)
                        
                else:
                    if i ==3:
                        clf = GenELMClassifier(hidden_layer=srhl_rbf)
                                   
                                    

        neigh_model = clf.fit(X[train_index], y[train_index])
        predictions = clf.predict(X[test_index])
        actuals = y[test_index]
        conf1=confusion_matrix(actuals, predictions)
        conf += conf1
        
        print(confusion_matrix(actuals, predictions))
    
    ax = sns.heatmap(conf,cmap="Wistia", annot=True, fmt='.0f')
    

    savetxt('data.csv', conf, delimiter=',')
    
    
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements 
    
    
    print(accuracy(conf))
    accu1 = accuracy(conf)
    
    
    
    
     ### nehigbors =16
    
    
    
    
     ######5 fold cross validation Clasificador ELM   final#########
     
     
    conf1=conf   
    
    def accuraciesNPM(conf1):
        accuracyNPM = 0;
        precisionNPM = 0;
        recallNPM = 0;
        specificityNPM = 0;
        F1scoreNPM = 0;
        for i in [0,1,2,3,4,5]:
          tpi=conf1[i,i]      
          fpi=0
          fni=0
          tni=0
          for k in [0,1,2,3,4,5]:
            if k != i:
              fpi += conf1[k,i]
              fni += conf1[i,k]
              for j in [0,1,2,3,4,5]:
                if j != i:
                  tni += conf1[k,j]
          accuracyNPM += (tpi+tni)/(tpi+tni+fni+fpi)/6 
          precisionNPM += tpi/(tpi+fpi)/6
          recallNPM += tpi/(tpi+fni)/6
          specificityNPM += tni/(tni+fpi)/6
        F1scoreNPM =2*precisionNPM*recallNPM/(precisionNPM + recallNPM)
        return accuracyNPM , precisionNPM , specificityNPM , recallNPM , F1scoreNPM
    
    confusionmia = accuraciesNPM(conf1)
    print(accuraciesNPM(conf1))   
    
    
    accu11 = accuraciesNPM(conf1)
    
    accu_veci = [*accu11]
    accu_veci.append(accu1)
    
    accu_veci = pd.DataFrame(accu_veci)
    accu_veci = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    
    # vectacc[i] = confusionmia[0]
    # vectacc[i+1]= vectacc[i]


    
    
    pickl = {'model': clf}
    pickle.dump(pickl, open('model_file'+'.p','wb'))
    
    
    file_name = 'model_file.p'
    with open(file_name,'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    
    #input transformed values to make a prediction in FlaskAPI
    model.predict(X[4,:].reshape(1,-1))
    model.predict(X[60,:].reshape(1,-1))
    
    model.predict(X_X_test[60,:].reshape(1,-1))
    
    
    predictionsegundoyear= model.predict(X_X_test)
    
    confsegundoyear=confusion_matrix(y_y_test, predictionsegundoyear)
    savetxt('datatest.csv', confsegundoyear, delimiter=',')
    #print(accuracy(confsegundoyear))
    confusionmia2 = accuraciesNPM(confsegundoyear)
    
    print(accuraciesNPM(confsegundoyear))
    
    accu2=accuracy(confsegundoyear)
    accu22 = accuraciesNPM(confsegundoyear)
    
    accu_veci_2 = [*accu22]
    accu_veci_2.append(accu2)
    
    accu_veci_2 = pd.DataFrame(accu_veci_2)
    accu_veci_2 = accu_veci.rename(index = {0:'accuracyNPM', 1:'precisiónNPM', 2:'specificidad',
                                                 3:'recallNPM', 4:'F1scoreNPM', 5:'Accuracy' })
    
    x = ['tanh', 'tribas', 'hardlim', 'rbf']
    accuracy_funcion_activacion [x[i]] = accu_veci
    accuracy_funcion_activacion_2[x[i]] = accu_veci_2
    
    vectacc2[i] = confusionmia2[0]
    # vectacc2[i+1]= vectacc2[i]
    
    
    
    
    del confusionmia
    del confusionmia2
    del X
    del y

accuracy_funcion_activacion = accuracy_funcion_activacion.transpose()
accuracy_funcion_activacion = accuracy_funcion_activacion.rename(index = {0:'tanh', 1:'tribas', 2:'hardlim', 3:'rbf'})


plt.plot(x,accuracy_funcion_activacion['Accuracy'])



################## plots #################################################

n=30
x=np.linspace(2,32,n,endpoint=False);
plt.plot(x,accuracy_vecinos['Accuracy'])

n=19
x=np.linspace(0.05,1,n,endpoint=False)
plt.plot(x,accuracy_min_dist['Accuracy'])

n=19
x=np.linspace(2,21,n,endpoint=False);
plt.plot(x,accuracy_n_comp['Accuracy'])

n=20
x=np.linspace(10,210,n,endpoint=False);
plt.plot(x,accuracy_nh['Accuracy'])

n=4
x = ['tanh', 'tribas', 'hardlim', 'rbf']
plt.plot(x,accuracy_funcion_activacion['Accuracy'])


accuracy_funcion_activacion_2 = accuracy_funcion_activacion_2.transpose()

accuracy_min_dist_2 = accuracy_min_dist_2.transpose()

accuracy_n_comp_2 = accuracy_n_comp_2.transpose()

accuracy_nh_2 = accuracy_nh_2.transpose()

accuracy_vecinos_2 = accuracy_vecinos_2.transpose()


tabla= pd.concat([accuracy_funcion_activacion, accuracy_funcion_activacion_2])

tabla.to_excel('tabla.xlsx')







    



