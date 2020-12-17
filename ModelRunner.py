import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

#class to handle reading/classifying/lagging the input data
class DataLoader:
    @staticmethod
    def __loadData(filePath):
        data = pd.read_csv(filePath, index_col=0)
        rets = (data.diff() / data).iloc[1:] #calculate simple returns

        return rets

    @staticmethod
    def __classifyData(rets, threshold):
        classified = rets.copy() #need to copy to avoid evaluating on already classified cells
        classified[rets > threshold] = 1 #positive movement
        classified[rets < -threshold] = -1 #negative movement
        classified.iloc[(rets >= -threshold) & (rets <= threshold)] = 0 #neutral movement

        return classified

    @staticmethod
    def __lagData(classified, lag): #lag the target column to avoid predicting with the same day's data
        lagged = classified.copy()
        lagged['SPY'][:-lag] = lagged['SPY'][lag:]
        lagged = lagged.iloc[1:]

        return lagged

    def getClassifiedData(self, filePath, threshold = 0.005, lag = 1): #wrapper of other functions
        rets = self.__loadData(filePath)
        classified = self.__classifyData(rets, threshold)
        lagged = self.__lagData(classified, lag)

        return lagged

#class that runs the Decision Tree and Random Forest
class ModelRunner:
    def __init__(self):
        self.xTrain = None #ETF training data
        self.xTest = None #ETF testing data
        self.yTrain = None #SPY training data
        self.yTest = None #SPY testing data
        # self.modelDict = {
        #     "MLPClassifier": MLPClassifier(),
        #     "KNN": KNeighborsClassifier(),
        #     "SVC": SVC(),
        #     "GaussianProcessClassifier": GaussianProcessClassifier(),
        #     "DecisionTree": DecisionTreeClassifier(),
        #     "RandomForest": RandomForestClassifier(),
        #     "AdaBoost": AdaBoostClassifier(),
        #     "NaiveBayes": GaussianNB(),
        #     "LDA": LinearDiscriminantAnalysis(),
        #     "QDA": QuadraticDiscriminantAnalysis()
        # }

    #calls the DataLoader class to get training and testing data
    def getData(self, filePath, threshold, testSize):
        data = DataLoader().getClassifiedData(filePath, threshold)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(data.iloc[:, 1:],
                                                                            data.iloc[:, 0],
                                                                            test_size=testSize)

    def runModels(self):
        #create a decision tree
        tree = DecisionTreeClassifier(criterion="gini", splitter="random")
        tree.fit(self.xTrain, self.yTrain)
        treePreds = tree.predict(self.xTest)
        print("Decision Tree Score:")
        print(tree.score(self.xTest, self.yTest))
        print("Decision Tree Confusion Matrix:")
        print(confusion_matrix(self.yTest, treePreds))

        #Hyperparameter tuning done with help from:
        #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        forest = RandomForestClassifier()
        # forestRandom = RandomizedSearchCV(estimator=forest, param_distributions=random_grid, n_iter=100, cv=3,
        #                                   random_state=0, verbose=2, n_jobs=-1)
        # forestRandom.fit(self.xTrain, self.yTrain)
        # bestRandomParams = forestRandom.best_params_

        #manually coded from best parameters of the random search
        #search all possible combinations
        param_grid = {
            'bootstrap': [True],
            'max_depth': np.arange(20, 40, 10),
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': np.arange(300, 500, 50)
        }
        forestGrid = GridSearchCV(estimator=forest, param_grid=param_grid,
                                  cv=3, n_jobs=-1, verbose=1)
        forestGrid.fit(self.xTrain, self.yTrain)

        bestForest = forestGrid.best_estimator_
        print("Random Forest Score:")
        print(bestForest.score(self.xTest, self.yTest))
        forestPreds = bestForest.predict(self.xTest)
        print("Random Forest Confusion Matrix:")
        print(confusion_matrix(self.yTest, forestPreds))

#self = ModelRunner()
#self.getData("Data/Spy_data.csv", 0.005, 0.2)

# scores = []
# for modelName in self.modelDict:
#     model = self.modelDict[modelName]
#     model.fit(self.xTrain, self.yTrain)
#     scores.append(model.score(self.xTest, self.yTest))
#     predictions = model.predict(self.xTest)
#
#     #print(modelName)
#     #print(model.score(xTest, yTest))
#     #print(confusion_matrix(yTest, predictions, labels=[-1, 0, 1]))
#
# return scores

# thresholds = np.arange(0.001, 0.006, 0.001)
# results = []
# for threshold in thresholds:
#     print(threshold)
#     runner = ModelRunner()
#     runner.getData("Data/SPY_data.csv", threshold, 0.2)
#     scores = runner.runModels()
#
#     counts = (runner.xTrain == 0).sum()
#     numNeutral = counts.sum()
#     total = runner.xTrain.count().sum()
#
#     results.append(scores > 1 - (numNeutral / total))
#
# resTable = pd.DataFrame(results)
# resTable.columns = runner.modelDict.keys()
# resTable = resTable.set_index(thresholds)
#
# for threshold in thresholds:
#     runner = ModelRunner()
#     runner.getData("Data/SPY_data.csv", threshold, 0.2)
#     counts = (runner.xTrain == 0).sum()
#     numNeutral = counts.sum()
#     total = runner.xTrain.count().sum()
#
#     print("Threshold: " + str(threshold))
#     print("% Neutral: " + str(numNeutral / total))
#     print("---------------")
