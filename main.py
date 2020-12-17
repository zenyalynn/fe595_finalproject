import pandas as pd
from SVM import svm_output
from KNN import KNN
from LDAQDA import runldaqda
from ModelRunner import ModelRunner


def main():
    # Used to run all of the respect ML algorithms on the data
    csv = "SPY_data.csv"
    stockdata = pd.read_csv(csv)
    print("   ----- Output for SVM Model -----   ", "\n")
    stockdata2 = stockdata.set_index(stockdata.columns[0])
    svm_output(stockdata2)
    print("\n", "   ----- Output for KNN Model -----   ", "\n")
    model1 = KNN(stockdata2)
    model1.perform_KNN()
    print("\n", "   ----- Output for LDA & QDA Model -----   ", "\n")
    runldaqda(stockdata)
    print("\n", "   ----- Output for Random Forest  -----   ", "\n")
    model2 = ModelRunner()
    #model2.getData(csv, 0.005, 0.2)
    #odel2.runModels()


if __name__ == "__main__":
    main()
