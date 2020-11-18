from .EvaluationData import EvaluationData
from .EvaluatedAlgorithm import EvaluatedAlgorithm


class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed

    def add_algorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def evaluate(self):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.get_name(), "...")
            results[algorithm.get_name()] = algorithm.evaluate(self.dataset)

        # Print results
        print("\n")

        print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
        for (name, metrics) in results.items():
            print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"],
                                                      metrics["MAE"]))

        print("\nLegend:\n")
        print(
            "RMSE:      Root Mean Squared Error. Lower values mean better accuracy."
        )
        print(
            "MAE:       Mean Absolute Error. Lower values mean better accuracy."
        )

    def sample_top_n(self, bx_book, test_subject=242, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.get_name())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.get_full_train_set()
            algo.get_algorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.dataset.get_anti_test_for_user(test_subject)

            predictions = algo.get_algorithm().test(testSet)

            recommendations = []

            print("\nWe recommend:")
            for user_id, item_id, actual_rating, estimated_rating, _ in predictions:
                recommendations.append((item_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:k]:
                print(bx_book.get_book_name(ratings[0]))
