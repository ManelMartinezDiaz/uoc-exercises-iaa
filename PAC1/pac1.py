from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import SVD

########Import IAA algorithms
from IAA_predictors import *

#############################
from surprise import Reader

def predict(data, uid, iid):

    trainset = data.build_full_trainset()
    
    # Set the algorithms to predict
    algorithms = (SVD, KNNBasic, KNNWithMeans, NormalPredictor)


    for a in algorithms:

        ########################
        # Init the algorithm and train it with trainset (fit it)
        # Store your prediction in a variable called "pred"
        algo = a()
        algo.fit(trainset)
        pred= algo.predict(uid,iid, r_ui=None, verbose=True)
      
        ########################

        print(pred)


def multiple_cv(data, folds=5):

    # Set the algorithms to cross validate
    algorithms = (SVD, KNNBasic, KNNWithMeans, NormalPredictor)
    measures = ['RMSE', 'MAE']


    for a in algorithms:
        ########################
        # Init the algorithm and perform the cross validation
        algo = a()

        cross_validate(algo, data, measures, cv=5, verbose=True, n_jobs=1)
        #cross_validate(algo, data, measures, folds=5, verbose=True)
        ########################


def run_IAAPredictors(data, do_cv=True, uid=-1, iid=-1):

    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    #####################################################################################
    # Use your fresh Dummy_IAA algorithm to get a prediction.

    algo = Dummy_IAA()

    if do_cv:
        # Run 5-fold cross-validation and print results.
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=1)
    else:
        if uid  > -1 and iid > -1:
            # Prepare the algorithm to get a prediction
            # Store your prediction in a variable called "pred"
            algo.fit(trainset)
            pred = algo.estimate(uid, iid)
            #pred = algo.predict(uid, iid,r_ui=None, verbose=True)
            
            print("Evaluating Dummy_IAA algorithm without cross-validation")
            print(pred)
        else:
            print("ERROR: if do_cv is False, uid and iid must be positive numbers")

    #####################################################################################
    # Use your fresh KNN_IAA algorithm to get a prediction.
    # Store your prediction in a variable called "pred"
    algo = KNN_IAA(n=35)
    
    if do_cv:
        # Run 5-fold cross-validation and print results.
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=1)
    else:
        if uid > -1 and iid > -1:
            # Prepare the algorithm to get a prediction
            # Store your prediction in a variable called "pred"
            algo.fit(trainset)
            pred = algo.estimate(uid, iid)
            #pred = algo.predict(uid, iid,r_ui=None, verbose=True)
            
            print("Evaluating KNN_IAA algorithm without cross-validation")
            print(pred)
        else:
            print("ERROR: if do_cv is False, uid and iid must be positive numbers")

##########################################################


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

##Set user id and item id
uid = 777
iid = 333


print("######################################")
print("##############Exercise 1##############")
print("######################################")

#Comment/uncomment to disable/run your method
predict(data,str(uid),str(iid))


print("######################################")
print("##############Exercise 2##############")
print("######################################")

#Comment/uncomment to disable/run your method
multiple_cv(data,folds=5)


print("######################################")
print("##############Exercise 4##############")
print("######################################")

#Comment/uncomment to disable/run your method
print("##############Do_cv=True##############")
#run_IAAPredictors(data,True, int(uid),int(iid))

print("##############Do_cv=False##############")
run_IAAPredictors(data,False, int(uid),int(iid))
