import numpy as np
import heapq
import random

from surprise import PredictionImpossible
from surprise import AlgoBase

class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.
    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. neighbor-based algorithms.
    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir #(item_inner_id, rating)
        self.yr = self.trainset.ir if ub else self.trainset.ur #(user_inner_id, rating)
        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

class Dummy_IAA(SymmetricAlgo):
    """Dummy IAA filtering algorithm.
    ---> Documentar mètode: args esperats, etc...
    """

    def __init__(self, sim_options={}, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)


    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        x, y = self.switch(u, i)

        est = 0
        #print("OK",u,i)

        ###############################
        ##compute and return estimation
        ##valoració a l’atzar d’entre totes les valoracions fetes per l’usuari que es consulta
        
        ##Fem un recorregut per self.yr (llista de tuples (user_inner_id, ratings)) 
        #de l'usuari u i es guarden només les valoracions a ListAux
        #S'agafa a l'atzar una valoració de la llista ListAux.
        ListAux = [(v) for k,v in self.yr[self.trainset.to_inner_uid(str(u))]]
        #print("ListAux:",ListAux)
        est = np.random.choice(ListAux)
        #est = 3.0

        return est


class KNN_IAA(SymmetricAlgo):
    """KNN IAA filtering algorithm.
    ---> Documentar mètode: args esperats, etc...
    """
    ##############################
    #######Complete __init__ method
    def __init__(self, k=40, min_k=1, sim_options={}, n=20, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.n = n

        ##################


        ##################
    ###############################

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        # Compute similarities between x and x2, where x2 describes all other
        # users that have also rated item Y.
        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        ####################################################################################

        #Compute weighted average from n random neighbors within k_neighbors
        #First initialize variables
        sum_sim = sum_ratings = actual_k = 0
        kn_length = len(k_neighbors)
       
        #Obtenim _random_neighbors (n_random_neighbors)
        #Si la mida de k_neigbors es inferior a n, n=kn_length
        if kn_length < self.n-1:
            n_random_neighbors = self.n
        else:
            #Sino seleccionem un número a l'atzar 
            #entre el mínim (self.n)i el màxim de neighbors (kn_length)
            n_random_neighbors = np.random.randint(self.n,kn_length+1)

        #Seleccionem a l'atzar n_random_neihbors de k_neighbors
        #print("neighbors",len(neighbors))
        #print("k_neighbors",len(k_neighbors))
        #print("n_random_neighbors",n_random_neighbors)
        #print("n",self.n)
        k_neighbors_random = random.sample(k_neighbors,n_random_neighbors)
        #print("n_random_neighbors:")
        #print("n_random_neighbors:",n_random_neighbors)
        #print("k_neighbors:",k_neighbors)
        #print("k_neighbors_random:",k_neighbors_random)
        
        #Calculem la mitja de k_neighbors_random
        for (sim, r) in k_neighbors_random:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1
                

        #####################

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details




