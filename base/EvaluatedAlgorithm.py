from RecommenderMetrics import RecommenderMetrics as rm

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def evaluate(self, evaluation_data,  verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluation_data.get_train_set())
        predictions = self.algorithm.test(evaluation_data.get_test_set())
        metrics["RMSE"] = rm.rmse(predictions)
        metrics["MAE"] = rm.mae(predictions)
        if (verbose):
            print("Analysis complete.")
    
        return metrics
    
    def get_name(self):
        return self.name
    
    def get_algorithm(self):
        return self.algorithm
    
    