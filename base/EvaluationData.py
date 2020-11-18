from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:
    
    def __init__(self, data, popularity_rankings, random_state=1):
        
        self.rankings = popularity_rankings
        
        #Build a full training set for evaluating overall properties
        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()
        
        #Build a 80/20 train/test split for measuring accuracy
        self.train_set, self.test_set = train_test_split(data, test_size=.20, random_state=random_state)
        
        #Build a "leave one out" train/test split for evaluating top-N recommenders
        #And build an anti-test-set for building predictions
        loo_cv = LeaveOneOut(n_splits=1, random_state=random_state)
        for train, test in loo_cv.split(data):
            self.loo_cv_train = train
            self.loo_cv_test = test
            
        self.loo_cv_anti_test_set = self.loo_cv_train.build_anti_testset()
        
        #Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.sim_algo = KNNBaseline(sim_options=sim_options)
        self.sim_algo.fit(self.full_train_set)
            
    def get_full_train_set(self):
        return self.full_train_set
    
    def get_full_anti_test_set(self):
        return self.full_anti_test_set
    
    def get_anti_test_for_user(self, test_subject):
        trainset = self.full_train_set
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def get_train_set(self):
        return self.train_set
    
    def get_test_set(self):
        return self.test_set
    
    def get_loo_cv_train_set(self):
        return self.loo_cv_train
    
    def get_loo_cv_test_set(self):
        return self.loo_cv_test
    
    def get_loo_cvAntiTestSet(self):
        return self.loo_cv_anti_test_set
    
    def get_similarities(self):
        return self.sims_algo
    
    def get_popularity_rankings(self):
        return self.rankings