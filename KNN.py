import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class KNN:
    def __init__(self, data):
        prices = data.iloc[1:]
        returns = prices.pct_change()
        returns = returns.iloc[1:]
        returns_model = returns.copy()
        self.dataset = returns_model

    def perform_KNN(self):
        tickers = ["SPY", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV", "XLF"]

        # Create matrix of values based on threshold
        for ticker in tickers:
            for i in range(0, len(self.dataset)):
                if self.dataset[ticker][i] < -0.005:
                    self.dataset[ticker][i] = -1
                elif self.dataset[ticker][i] > 0.005:
                    self.dataset[ticker][i] = 1
                else:
                    self.dataset[ticker][i] = 0

        # Define predictors and target, and lag SPY returns by 1 day
        predictors = self.dataset[["XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV", "XLF"]]
        target = self.dataset["SPY"]
        predictors = predictors.iloc[:len(predictors) - 1]
        target = target.iloc[1:]
        target = np.asarray(target, dtype="|S6")

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=6)

        # Make list of accuracy rates and plot to see which k to choose
        k_range = range(1, 26)
        scores = {}
        scores_list = []
        for k in k_range:
            # Create KNN Classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            # Train the model using the training set
            knn_model = knn.fit(X_train, y_train)
            # Predict the response for test set
            y_pred = knn_model.predict(X_test)
            # Model Accuracy
            scores[k] = metrics.accuracy_score(y_test, y_pred)
            scores_list.append(metrics.accuracy_score(y_test, y_pred))

        plt.plot(k_range, scores_list)
        plt.xlabel("Value of k for KNN")
        plt.ylabel("Testing Accuracy")
        plt.show()

        # Re-train model for k=20 because it had the best result
        knn = KNeighborsClassifier(n_neighbors=19)
        knn_model1 = knn.fit(X_train, y_train)
        y_pred = knn_model1.predict(X_test)
        print("Accuracy score for KNN Model:", metrics.accuracy_score(y_test, y_pred))
        # Accuracy: 0.5168986083499006
        print("Confusion Matrix for KNN Model:", "\n", confusion_matrix(y_test, y_pred))

