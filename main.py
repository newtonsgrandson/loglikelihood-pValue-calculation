import random
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
import statsmodels.api as sm

#salary is y column name for this data
yLabel = "salary"

def newTableMain():
    def fillNullItems(column):
        def generateNewEl(column, idx):
            uniqueList = list(column.unique())
            uniqueList.remove(" ?")
            idx = random.randint(0, uniqueList.__len__() - 1)
            return uniqueList[idx]

        newCol = [column[i] if str(column[i]) != " ?" else generateNewEl(
            column, i) for i in range(column.__len__())]
        return newCol

    #Data label editting for this data
    table = pd.read_csv("adult.csv")
    table = table.loc[:, ["1", "3", "5", "6", "7", "8", "9", "13", "14"]]
    labels = ["workClass", "education", "martialStatus",
              "occupation", "relationship", "race", "sex",
              "nativeCountry", "salary"]

    table.columns = labels
    for i in table.columns:
        table.loc[:, i] = fillNullItems(table.loc[:, i])

    table.to_csv("data.csv")

def labelEncoder(column):
    values = column.values
    unique = list(column.unique())
    name1 = column.name

    encoded = [unique.index(i) for i in values]
    return pd.Series(encoded, name=name1)

def tablePreprocessing(table):
    def cathegoricalToBin(column):
        column = pd.Series(column)
        targets = column.unique()
        column = labelEncoder(column)

        newCol = []
        for i in column:
            row = [1 if i == j else 0 for j in range(targets.__len__())]
            newCol.append(row)

        return pd.DataFrame(newCol, columns=[f"{column.name}{i}" for i in targets])

    newTable = pd.DataFrame()
    if type(table) == type(pd.DataFrame()):
        for i in table.columns:
            if str(i) == "salary":
                newTable = pd.concat(
                    [newTable, labelEncoder(table.loc[:, i])], axis=1)
            else:
                newTable = pd.concat([newTable, cathegoricalToBin(
                    labelEncoder(table.loc[:, i]))], axis=1)
                
    elif type(table) == type(pd.Series()):
        newTable = pd.concat([newTable, cathegoricalToBin(labelEncoder(table))], axis=1) 
    
    else:
        raise("Table tipi ya DataFrame olmalı ya Series.")
        
    return newTable

def rowPreprocessing(table, row):
    features = list(row.keys())
    values = list(row.values())

    ROW = []
    for i in range(features.__len__()):
        column = table.loc[:, features[i]]
        unique = list(column.unique())
        idx = unique.index(values[i])
        value = [1 if i == idx else 0 for i in range(unique.__len__())]
        ROW.append(value)
    
    ROW = pd.Series([j for i in ROW for j in i], name = 0)
    ROW = {"0": ROW}
    ROW = pd.DataFrame(ROW).transpose()
    return ROW

def likehoodValue(keyword, table):
    global yLabel
    X = tablePreprocessing(table.loc[:, keyword])
    y = labelEncoder(table.loc[:, yLabel])
    
    ct = CategoricalNB()
    ct.fit(X, y)
    statModel = sm.OLS(ct.predict(X), X)
    results = statModel.fit()
    results_summary = results.summary()
    results_as_html = results_summary.tables[0].as_html()
    stats = pd.read_html(results_as_html, header=0, index_col=0)[0]
    logLikehood = stats.loc['Time:','1.000']
    print(f"{keyword} Log-Likelihood value: {logLikehood}")

def predictATKeywords(rowDict, table):
    global yLabel
    features = list(rowDict.keys())
    row = rowPreprocessing(table, rowDict)
    X = tablePreprocessing(table.loc[:, features])
    y = labelEncoder(table.loc[:, yLabel])
    
    ct = CategoricalNB()
    ct.fit(X, y)
    statModel = sm.OLS(ct.predict(X), X)
    results = statModel.fit()
    results_summary = results.summary()
    results_as_html = results_summary.tables[1].as_html()
    stats = pd.read_html(results_as_html, header=0, index_col=0)[0]
    priorValues = stats.loc[:, "P>|t|"].unique()
    priorValues = pd.Series(priorValues, index = features, name = "Prior Values")
    
    prediction = ct.predict(row)
    print("Örnek satırın tahmin değeri:", table.loc[:, yLabel].unique()[prediction][0])
    print(priorValues)
    return priorValues        

newTableMain()
data = pd.read_csv("data.csv", index_col=0)
likehoodValue("workClass", data)

print("Prior Değerler ve tahmin")
sampleRow = {"workClass": " Private",
             "education": " Bachelors",
             "martialStatus": " Never-married",
             "occupation": " Armed-Forces",
             "relationship": " Own-child",
             "race": " White",
             "sex": " Female",
             "nativeCountry": " Iran"
             }
predictATKeywords(sampleRow, data)