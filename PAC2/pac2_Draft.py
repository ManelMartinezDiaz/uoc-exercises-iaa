import pandas as pd
import numpy
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#from sklearn.tests.test_multiclass import n_classes

#import pac2_classifier_comparison as cc

def loadAndPreprocess(filename):
    # ------------------------------------------------------------------------------------------------
    print("Activity 1a: ")

    dataLabels = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'class'
    ]

      
    data=pd.read_csv(filename,sep=',',header=None,names=dataLabels,na_values=["?"])
   
    #print("Original data")
    #print(data.ix[:,0:])
   
    n=len(data)
    #print ("Amount of instances: " + str(n))
    #print ("Analyzing class distribution")
    #print (list(data['class'].value_counts()))
    
    #print("Rows with missing values")
    #print(sum(numpy.isnan(data).any(axis=1)))
    #print("Attributes with missing values")
    #print(len(data.isnull().sum().loc[data.isnull().sum()> 0]))
    

    #Remove rows with missing data
    cleanData=data[~numpy.isnan(data).any(axis=1)]
    cleanData=cleanData.reset_index(drop=True)  #Required. Otherwise, the index of the rows dropped keep active

    #print("Clean data")
    #print(cleanData[:])
    
    
    # Separating classes (Y) from values (X)
    dataX=cleanData.ix[:,0:13]
    dataY=cleanData.ix[:,13]


    # Extract status and standardize product values
    attributes = preprocessing.scale(dataX)
    #print("Scaled data")
    #print(attributes[:])
    return attributes, dataY
    
def exercise1(attributes, classes):
    print("Activity 1a")
    # Apply PCA requesting all components (no argument)
    pca_1a = PCA()
    pca_1a.fit(attributes)
    
    #Variancia explicada acumulada per cada component
    print("Resultat 1a:")
    print(numpy.cumsum(pca_1a.explained_variance_ratio_))
    
    #Calcular pèrdua d'informació amb 2,4 i 8 components respecte a original.
    print("Activity 1b")
    pca_1b = PCA()
    pca_1b.fit(attributes,classes)
    X_pca_1b = pca_1b.transform(attributes)
    XI_pca_1b = pca_1b.inverse_transform(X_pca_1b)
    
    pca_1b_n2 = PCA(n_components=2)
    pca_1b_n2.fit(attributes,classes)
    X_pca_1b_n2 = pca_1b_n2.transform(attributes)
    XI_pca_1b_n2 = pca_1b_n2.inverse_transform(X_pca_1b_n2)

    print("Resultat 1b:")  
    mse_1b_n2 = mean_squared_error(XI_pca_1b, XI_pca_1b_n2)
    print("MSE amb 2 Components", mse_1b_n2)
    
    pca_1b_n4 = PCA(n_components=4)
    pca_1b_n4.fit(attributes,classes)
    X_pca_1b_n4 = pca_1b_n4.transform(attributes)
    XI_pca_1b_n4 = pca_1b_n4.inverse_transform(X_pca_1b_n4)

    mse_1b_n4 = mean_squared_error(XI_pca_1b, XI_pca_1b_n4)
    print("MSE amb 4 Components", mse_1b_n4)

    
    pca_1b_n8 = PCA(n_components=8)
    pca_1b_n8.fit(attributes,classes)
    X_pca_1b_n8 = pca_1b_n8.transform(attributes)
    XI_pca_1b_n8 = pca_1b_n8.inverse_transform(X_pca_1b_n8)

    mse_1b_n8 = mean_squared_error(XI_pca_1b, XI_pca_1b_n8)
    print("MSE amb 8 Components", mse_1b_n8)

    print("Activity 1c")
    print("Resultat 1c:")
    fig = plt.figure()
    fig.suptitle('Dades originals 2 primers components')
    plt.scatter(attributes[:,0], attributes[:,1], marker='o',c=classes)
    plt.show()
    
    fig = plt.figure()
    fig.suptitle('Dades transformades 2 primers components')
    plt.scatter(X_pca_1b_n2[:,0], X_pca_1b_n2[:,1], marker='o',c=classes)
    plt.show()

    
def exercise2(attributes, classes):
    # ------------------------------------------------------------------------------------------------
    print("Activity 2")
    
    ### Activity 2a ####
    k = 3
    n_splits = 5
    score_source = []
    
    kf = KFold(n_splits)
    kf.split(attributes)
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean') 
                
    for X_train, X_test in kf.split(attributes):
       x_train,x_test = numpy.array(attributes)[X_train], numpy.array(attributes)[X_test]
       y_train,y_test = numpy.array(classes)[X_train],numpy.array(classes)[X_test]
       
       knn.fit(x_train, y_train)
       score_source.append(knn.score(x_test, y_test))
       
    print("2.a: Score Conjunt Original:", numpy.mean(score_source))

    ### Activity 2b ####       
    score_transformada = []
    
    pca_n8 = PCA(n_components=8)
    pca_n8.fit(attributes,classes)
    X_pca_n8 = pca_n8.transform(attributes)
        
    for X_train, X_test in kf.split(attributes):
       xt_train,xt_test = numpy.array(X_pca_n8)[X_train],numpy.array(X_pca_n8)[X_test]
       y_train,y_test = numpy.array(classes)[X_train], numpy.array(classes)[X_test]
              
       knn.fit(xt_train, y_train)
       score_transformada.append(knn.score(xt_test, y_test))

    print("2.b: Score Conjunt Transformat:", numpy.mean(score_transformada))
    
    
def exercise3(attributes, classes):
    print("Activity 3")

    n_splits = 3
    neighbours = (3,4,5)
    knn_n3_score = []; knn_n3_train_time = []; knn_n3_predict_time = []
    knn_n4_score = []; knn_n4_train_time = []; knn_n4_predict_time = []
    knn_n5_score = []; knn_n5_train_time = []; knn_n5_predict_time = []
    svc_score = []; svc_train_time = []; svc_predict_time = []
    dtc_score = []; dtc_train_time = []; dtc_predict_time = []
    abc_score = []; abc_train_time = []; abc_predict_time = []
    gnb_score = []; gnb_train_time = []; gnb_predict_time = []
    
    kf = KFold(n_splits)
    
    count_split = 1
    
    for X_train, X_test in kf.split(attributes):
    
       x_train,x_test = numpy.array(attributes)[X_train],numpy.array(attributes)[X_test]
       y_train,y_test = numpy.array(classes)[X_train], numpy.array(classes)[X_test]
       
       aux= 0

       for k in neighbours:
          
          #print("k Nearest Neighbors",neighbours[aux])
          clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
          start = time.clock()
          clf.fit(x_train, y_train)
          train_time = time.clock() - start
          start = time.clock()
          y_pred = clf.predict(x_test)
          predict_time = time.clock() - start
          score = accuracy_score(y_test, y_pred)
       
          if k == 3:
               knn_n3_score.append(score)
               knn_n3_train_time.append(train_time) 
               knn_n3_predict_time.append(predict_time)
          if k == 4:
               knn_n4_score.append(score)
               knn_n4_train_time.append(train_time) 
               knn_n4_predict_time.append(predict_time)
          if k == 5:
               knn_n5_score.append(score)
               knn_n5_train_time.append(train_time) 
               knn_n5_predict_time.append(predict_time)
          aux += 1
          
       ### Linear SVM ###  SVC()     
       #print("SVC")             
       clf = SVC(kernel='linear',C=0.025)         
       start = time.clock()
       clf.fit(x_train, y_train)
       train_time = time.clock() - start
       start = time.clock()
       y_pred = clf.predict(x_test)
       predict_time = time.clock() - start
       score = accuracy_score(y_test, y_pred)
       
       svc_score.append(score)
       svc_train_time.append(train_time) 
       svc_predict_time.append(predict_time)
       
        
       ### Decission Tree ###     DecisionTreeClassifier()     
       #print("Decission Tree")             
       clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)         
       start = time.clock()
       clf.fit(x_train, y_train)
       train_time = time.clock() - start
       start = time.clock()
       y_pred = clf.predict(x_test)
       predict_time = time.clock() - start
       score = accuracy_score(y_test, y_pred)
       
       dtc_score.append(score)
       dtc_train_time.append(train_time) 
       dtc_predict_time.append(predict_time)
           
       ### AdaBoost ####    AdaBoostClassifier()    
       #print("AdaBoost")             
       clf = AdaBoostClassifier()         
       start = time.clock()
       clf.fit(x_train, y_train)
       train_time = time.clock() - start
       start = time.clock()
       y_pred = clf.predict(x_test)
       predict_time = time.clock() - start
       score = accuracy_score(y_test, y_pred)
       
       abc_score.append(score)
       abc_train_time.append(train_time) 
       abc_predict_time.append(predict_time)
       
       ### Gaussian Naive Bayes ####   GaussianNB()   
       #print("Gaussian Naive Bayes")             
       clf = GaussianNB()         
       start = time.clock()
       clf.fit(x_train, y_train)
       train_time = time.clock() - start
       start = time.clock()
       y_pred = clf.predict(x_test)
       predict_time = time.clock() - start
       score = accuracy_score(y_test, y_pred)
       
       gnb_score.append(score)
       gnb_train_time.append(train_time) 
       gnb_predict_time.append(predict_time)
       
       count_split += 1
       
    
    #FOR finalitzat Mostrar resultats
    print("Score Result")
    print(numpy.mean(knn_n3_score),"KNN 3 Neigbours")
    print(numpy.mean(knn_n4_score),"KNN 4 Neigbours")
    print(numpy.mean(knn_n5_score),"KNN 5 Neigbours")
    print(numpy.mean(svc_score),"Linear SVM")
    print(numpy.mean(dtc_score),"Decission Tree")
    print(numpy.mean(abc_score),"AdaBoost")
    print(numpy.mean(gnb_score),"Gaussian Naive Bayes")
    
    print("Training Time")
    print(numpy.mean(knn_n3_train_time),"KNN 3 Neigbours")
    print(numpy.mean(knn_n4_train_time),"KNN 4 Neigbours")
    print(numpy.mean(knn_n5_train_time),"KNN 5 Neigbours")
    print(numpy.mean(svc_train_time),"Linear SVM")
    print(numpy.mean(dtc_train_time),"Decission Tree")
    print(numpy.mean(abc_train_time),"AdaBoost")
    print(numpy.mean(gnb_train_time),"Gaussian Naive Bayes")
    
    print("Prediction Time")
    print(numpy.mean(knn_n3_predict_time),"KNN 3 Neigbours")
    print(numpy.mean(knn_n4_predict_time),"KNN 4 Neigbours")
    print(numpy.mean(knn_n5_predict_time),"KNN 5 Neigbours")
    print(numpy.mean(svc_predict_time),"Linear SVM")
    print(numpy.mean(dtc_predict_time),"Decission Tree")
    print(numpy.mean(abc_predict_time),"AdaBoost")
    print(numpy.mean(gnb_predict_time),"Gaussian Naive Bayes")    
          

    
def exercise4(attributes, classes):
    
    print("Activity 4")

    #Exercici 4a

    n_splits = 3
    accuracy_scores = []
    precision_scores = []
    
    kf = KFold(n_splits) 
    
    for X_train, X_test in kf.split(attributes):
    
       x_train,x_test = numpy.array(attributes)[X_train],numpy.array(attributes)[X_test]
       y_train,y_test = numpy.array(classes)[X_train],numpy.array(classes)[X_test]

       ### Linear SVM ###  SVC()     
       #print("SVC")             
       clf = SVC(kernel='linear',C=0.025)         
       clf.fit(x_train, y_train)
       y_pred = clf.predict(x_test)
       svc_accuracy_score = accuracy_score(y_test, y_pred)
       svc_precision_score = precision_score(y_test, y_pred, average='macro')
       
       accuracy_scores.append(svc_accuracy_score)
       precision_scores.append(svc_precision_score)
       
    print("4.a")
    print("Linear SVM")
    print(numpy.mean(accuracy_scores),"Accuracy Score")
    print(numpy.mean(precision_scores),"Precision Score")

    #Exercici 4b
   
    accuracy_scores = []
    precision_scores = []
    
    skf = StratifiedKFold(n_splits) 
    
    for X_train, X_test in skf.split(attributes,classes):
    
       x_train,x_test = numpy.array(attributes)[X_train],numpy.array(attributes)[X_test]
       y_train,y_test = numpy.array(classes)[X_train],numpy.array(classes)[X_test]

       ### Linear SVM ###  SVC()     
       #print("SVC")             
       clf = SVC(kernel='linear',C=0.025)         
       clf.fit(x_train, y_train)
       y_pred = clf.predict(x_test)
       
       svc_accuracy_score = accuracy_score(y_test, y_pred)
       svc_precision_score = precision_score(y_test, y_pred, average='macro')
       
       accuracy_scores.append(svc_accuracy_score)
       precision_scores.append(svc_precision_score)
       
    print("4.b") 
    print("Linear SVM")
    print(numpy.mean(accuracy_scores),"Accuracy Score")
    print(numpy.mean(precision_scores),"Precision Score")
    
    
# MAIN
 
X, y = loadAndPreprocess('processedCleveland.csv')

exercise1(X, y)

exercise2(X, y)

exercise3(X, y)

exercise4(X, y)


print("Fi")