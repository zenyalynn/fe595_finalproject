import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


## ldaqda performs LDA and QDA with the training data and finds the accuracy of the test data of each model
## prints the accuracy of LDA and QDA of testing data
def ldaqda(x_train, x_test, y_train, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train,y_train)
    predictedvalues = lda.predict(x_test)
    y_test = y_test.reset_index(drop =True)
    predictedvalues = pd.Series(predictedvalues)
    print(np.mean(predictedvalues==y_test))


    qda = QuadraticDiscriminantAnalysis()
    qdafit = qda.fit(x_train,y_train)
    predictedvalues2 = qdafit.predict(x_test)
    predictedvalues2 = pd.Series(predictedvalues2)
    print(np.mean(predictedvalues2==y_test))

## LDAtesting tries to find the highest accuracy for the LDA model by varying different shrinkage values with type lsqr and eigen
## outputs table with all accuracies with shrinkage value and lists max accuracy and where it occurs
def ldatesting(x_train, x_test, y_train, y_test):
    numlist = np.arange(0.1, 1, 0.05).tolist()
    typelist = ['lsqr']*18 + ['eigen']*18
    numlistfin = numlist + numlist
    lsqracc = []
    for j in range(0,len(numlist)):
        lda2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=numlist[j],n_components=2)
        lda2.fit(x_train, y_train)
        predictedvalues2 = lda2.predict(x_test)
        predictedvalues2 = pd.Series(predictedvalues2)
        y_test = y_test.reset_index(drop=True)
        lsqracc = lsqracc + [np.mean(predictedvalues2 == y_test)]

    eigenacc = []
    for j in range(0,len(numlist)):
        lda2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=numlist[j],n_components=2)
        lda2.fit(x_train, y_train)
        predictedvalues2 = lda2.predict(x_test)
        predictedvalues2 = pd.Series(predictedvalues2)
        y_test = y_test.reset_index(drop=True)
        eigenacc = eigenacc + [np.mean(predictedvalues2 == y_test)]

    acc = lsqracc + eigenacc
    ldadf = pd.DataFrame(list(zip(typelist,numlistfin,acc)), columns = ['Type','Shrinkage','Accuracy'])
    print(ldadf)
    print('Max')
    print(ldadf.iloc[ldadf['Accuracy'].idxmax()])

##runldaqda creates testing and training data of the SPY_data.csv file
def runldaqda(pd1):
    for i in range(1, 11):
        pd1[pd1.columns[i]] = pd1[pd1.columns[i]].diff()

    pd1 = pd1.iloc[1:]
    pd2 = pd1.drop(pd1.columns[0], axis=1)

    for index, row in pd2.iterrows():
        for i in range(0, 10):
            if (row[i] > 0.0005):
                row[i] = 1
            if (row[i] < -0.0005):
                row[i] = -1
            if (row[i] <= 0.0005 and row[i] >= -0.0005):
                row[i] = 0

    lagged = pd2.copy()
    lagged['SPY'][:-1] = lagged['SPY'][1:]
    lagged = lagged.iloc[1:]

    pd2target = lagged[lagged.columns[0]]
    pd2data = lagged.drop(lagged.columns[0], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(pd2data, pd2target, test_size=0.3)
    ldaqda(x_train, x_test, y_train, y_test)
    ldatesting(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    pd1 = pd.read_csv('SPY_data.csv')
    runldaqda(pd1)

