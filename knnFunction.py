##This code sample is part of a project which tries to optimize the stock industry classification.
import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier


def gettoday():
    return int(datetime.date.today().isoformat().replace("-",""))


def getfinite(data):
    numberdata = data.select_dtypes(include=[np.number])
    numberdata = numberdata[np.isfinite(numberdata)].dropna().reset_index()
    data = pd.merge(data.reset_index(), numberdata).drop("index", axis=1)
    return data


def calknnabnormal(finaldata, knnthreshold,quarterdate):
    abnormal = list()
    for i in range(len(quarterdate)-1):
        quarterdata = finaldata[finaldata["date"] > quarterdate[i]]
        quarterdata = quarterdata[quarterdata["date"] <= quarterdate[i+1]]
        # Obtain quarterly frequency data
        tmpdata = quarterdata.groupby("code").apply(lambda x: len(x["induscode"].unique())).reset_index()
        # tmpdata is the dataframe of the number of industries that all individual stocks belong to in the current quarter
        singledata = quarterdata[quarterdata["code"].isin(tmpdata[tmpdata[0] == 1]["code"])]
        singledata = singledata[["code", "induscode", "indusname"]].drop_duplicates()
        singledata = pd.merge(singledata, singledata.groupby("code").agg({"indusname": max}).reset_index())
        singledata["quarter"] = quarterdate[i+1]
        singledata["type"] = 1
        abnormal.extend(singledata[["code", "quarter", "induscode", "indusname", "type"]].values.tolist())
        quarterdata = quarterdata[quarterdata["code"].isin(tmpdata[tmpdata[0] != 1]["code"])]
        for j in quarterdata["code"].unique():
            tmpdata=quarterdata[np.in1d(quarterdata["code"],j)]
            tmp=pd.DataFrame(tmpdata["induscode"].value_counts()).reset_index().sort_values("induscode")
            if(tmp["induscode"].iloc[-1]/tmp["induscode"].sum()>knnthreshold):
                abnormal.append([j,quarterdate[i+1],tmp["index"].iloc[-1],tmpdata[tmpdata["induscode"]==tmp["index"].iloc[-1]]["indusname"].iloc[-1],1])
            else:
                tmpindus=tmpdata[["induscode","indusname"]].drop_duplicates()
                abnormal.append([j,quarterdate[i+1],tmpindus["induscode"].tolist(),tmpindus["indusname"].tolist(),2])
        #Abnormal refers to the stocks whose industry assignment changes within a quarter, 
        #and the leading industry frequency is less than knnthreshold.
        #Normalrefers to the stocks whose industry assignment changes within a quarter, 
        #and the leading industry frequency is larger than knnthreshold.
    abnormal=pd.DataFrame(abnormal)
    abnormal.columns=["code","date_period","induscode","indusname","type"]
    return abnormal


def train_test_split(X,Y,test_ratio=0.15,seed=None):
    if seed:
        np.random.seed(seed)
    shuffled_index = np.random.permutation(len(X))
    test_size = int(len(X)*test_ratio)
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]
 
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]
    return X_train,X_test,Y_train,Y_test


def calknnresult(data, abnormal, final_train, varnames,quarterdate,neighbor=5,test_ratio=0.15):
    knnresult=list()
    accuracy=list()
    for i in range(1,len(quarterdate)):
        predict_data=pd.merge(data,abnormal[abnormal["date_period"]==quarterdate[i]])
        predict_data=getfinite(predict_data)
        for j in range(len(predict_data)):
            tmpdata=predict_data.iloc[j]
            X_predict=tmpdata[varnames].values
            X_indus=tmpdata["induscode"]
            #X_indus is a list of all the industries that the stock belongs to in a quarter
            tmp_train=final_train[final_train["induscode"].isin(X_indus)]
            tmp_train=tmp_train[tmp_train["date_period"]<=quarterdate[i]]
            #tmp_train=tmp_train[tmp_train["report_period"]>=quarterdate[i]-10000]
            #Limit last year
            Y_train=tmp_train["induscode"].values
            X_train=tmp_train[varnames].values
            X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_ratio)
            #Combined with KNN accuracy performance, 
            ##only all industries of the current abnormal stocks were trained as training sets
            if(len(X_train)==0):
                continue
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(X_train,Y_train.astype(int))
            knnindus=knn.predict(X_predict.reshape(1, -1))[0]
            test=knn.predict(X_test)
            accuracy.append([tmpdata["code"],tmpdata["date_period"],tmpdata["induscode"],sum(test==Y_test)/len(Y_test)])
            knnresult.append([tmpdata["code"],tmpdata["date_period"], knnindus,tmpdata["indusname"][tmpdata["induscode"].index(knnindus)]])
    knnresult=pd.DataFrame(knnresult)
    knnresult.columns=["code","date_period","induscode","indusname"]
    accuracy=pd.DataFrame(accuracy)
    accuracy.columns=["code","date_period","induscode","accuracy"]
    #accuracy.to_excel("output/accuracy.xlsx")
    return knnresult,accuracy

def accuracyoutput(alldata,accuracy):
    swindusname=alldata[["swcode","swname"]].drop_duplicates().groupby("swcode").agg({"swname":max}).reset_index()
    zxindusname=alldata[["zxcode","zxname"]].drop_duplicates().groupby("zxcode").agg({"zxname":max}).reset_index()
    swindusname.columns=["induscode","indusname"]
    zxindusname.columns=["induscode","indusname"]
    indusname=pd.concat([swindusname,zxindusname])
    induscode=indusname["induscode"].values.tolist()
    indusname=indusname["indusname"].values.tolist()
    tmpname=list()
    for i in range(len(accuracy)):
        tmplist=list()
        for j in accuracy["induscode"].iloc[i]:
            tmplist.append(indusname[induscode.index(j)])
        tmpname.append(tmplist)
    accuracy["indusname"]=pd.Series(tmpname)
    accuracy.to_excel("output/accuracy.xlsx")


def diffoutput(knnresult,alldata,daydf,quarterdate):
    # Output the information of stocks whose industry has changed on a quarterly basis
    lastknn = knnresult.groupby("code").apply(lambda x:x[["induscode","indusname"]].shift()).reset_index()
    lastknn = lastknn.rename({"induscode":"lastcode","indusname":"lastname"},axis=1)
    diffknn = pd.merge(knnresult.reset_index(),lastknn)
    diffknn = diffknn.dropna().drop("index",axis = 1)
    diffknn = diffknn[diffknn["induscode"]!=diffknn["lastcode"]]
    diffknn = pd.merge(daydf,diffknn)
    diffknn = pd.merge(alldata,diffknn).drop(["pctchg","floatmarketcap","const"],axis=1)
    diffknn = diffknn.groupby(["date_period","code"]).agg({"swcode":"unique","swname":"unique","zxcode":"unique",
    "zxname":"unique","induscode":"unique","indusname":"unique","lastcode":"unique","lastname":"unique","type":"unique"}).reset_index()
    rawdiff = diffknn.copy()
    diffknn["last_period"] = diffknn["date_period"].apply(lambda x:quarterdate[quarterdate.index(x)-1])
    diffknn = pd.merge(daydf.rename({"date_period":"last_period"},axis = 1),diffknn)
    diffknn = pd.merge(diffknn,alldata.rename({"swcode":"lastswcode","swname":"lastswname",
                                 "zxcode":"lastzxcode","zxname":"lastzxname"},axis=1))
    diffknn = diffknn.drop(["pctchg","floatmarketcap","const"],axis=1)
    diffknn = diffknn.groupby(["date_period","code"]).agg({"lastswcode":"unique","lastswname":"unique",
                                         "lastzxcode":"unique","lastzxname":"unique"}).reset_index()
    for i in ["zxcode","swcode","zxname","swname","induscode","indusname","lastcode","lastname","type"]:
        rawdiff[i] = rawdiff[i].apply(lambda x:list(map(lambda y:str(y),x)))
        rawdiff[i] = rawdiff[i].apply(lambda x:",".join(x))
    for i in ["lastzxcode","lastzxname","lastswcode","lastswname"]:
        diffknn[i] = diffknn[i].apply(lambda x:list(map(lambda y:str(y),x)))
        diffknn[i] = diffknn[i].apply(lambda x:",".join(x))
    diffknn = pd.merge(diffknn,rawdiff,how = "right")
    diffknn.to_excel("output/diffknn.xlsx")