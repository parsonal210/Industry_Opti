import pandas as pd
import numpy as np
from abnormalLoad import getdata, getknndata
from abnormalFunction import calpreknnindus,outputdiffr2
from knnFunction import gettoday,calknnabnormal, getfinite, calknnresult, diffoutput,accuracyoutput
from newindusFunction import caldaydf, calindusprice, indusoutput


def getnewindus(maxwindow=120, minwindow=15, industype="Lv1", knnthreshold=0.75):
    data = getknndata()
    # getknndata获取KNN模型所需的数据
    quarterdate = sorted(data["date_period"].unique()) 
    corrdate = quarterdate.copy()
    corrdate.extend([20131231,20191129,20210729,gettoday()])
    #日期左开右闭
    corrdate = sorted(corrdate)
    corrdict=dict()
    for i in range(1,len(corrdate)):
        corrdict[i]=[corrdate[i-1],corrdate[i]]
    varnames = ["assetturnover", "profitmargin", "debt2AssetRatio", "ROA", "ROE", "ROIC", "revenue_MA_growth", "NP_MA_growth"]
    # varnames为KNN模型中使用到的财务数据变量
    alldata = getdata(industype)
    # getdata获取回归计算拟合优度所需数据
    finaldata = calpreknnindus(alldata, corrdict, maxwindow, minwindow)
    # calpreknnindus回归得到初步行业分类
    abnormal = calknnabnormal(finaldata, knnthreshold,quarterdate)
    # 按照knnthreshold得到需要KNN处理的异常个股abnormal
    # 以及不需要KNN处理的正常个股normal，直接输出
    normal = abnormal[abnormal["type"] == 1].drop("type", axis=1)
    abnormal = abnormal[abnormal["type"] == 2].drop("type", axis=1)   
    abnormal.to_excel("output/abnormal.xlsx")
    final_train = getfinite(pd.merge(normal, data))
    # 处理得到KNN训练数据
    # 处理得到KNN训练数据
    knnresult,accuracy = calknnresult(data, abnormal, final_train, varnames,quarterdate,5)
    accuracyoutput(alldata,accuracy)
    normal["type"] = "R2"
    knnresult["type"] = "KNN"
    # data+abnormal得到predict data
    # calknnresult以normal的数据作为训练集，varnames为变量对异常个股进行预测
    knnresult = pd.concat([normal, knnresult]).reset_index(drop=True).sort_values("date_period")
    # knn处理结果与normal合并得到最终季度频率行业分类
    daydf = caldaydf(alldata,quarterdate)
    #获取三个报告期内的工作日
    diffoutput(knnresult,alldata,daydf,quarterdate)
    #输出行业归属发生变化的记录
    accuracy = pd.read_excel("output/accuracy.xlsx",index_col=0)
    accuracy = accuracy.rename({"induscode":"knninduscode","indusname":"knnindusname"},axis=1)
    diffknn = pd.read_excel("output/diffknn.xlsx",index_col=0)
    pd.merge(diffknn,accuracy,how="left").to_excel("output/diffknn.xlsx")
    outputdiffr2(alldata,data,daydf)
    diffknn = pd.read_excel("output/diffknn.xlsx",index_col=0)
    diffknn["last_period"] = diffknn["date_period"].apply(lambda x:quarterdate[quarterdate.index(x)-1])
    abnormal = abnormal.rename({"date_period":"last_period"},axis=1)[["code","last_period"]]
    abnormal["last_type"] = "KNN"
    pd.merge(abnormal,diffknn,how="right").to_excel("output/diffknn.xlsx")
    finaldata = pd.merge(knnresult, daydf)
    finaldata["induscode"] = finaldata["induscode"].astype(float)
    indusoutput(finaldata, industype)
    # 将日频数据输出为mat格式
    calindusprice(industype, calmindata=False)
    # 读取mat格式行业分类数据，计算开收盘价与amount，默认不计算分钟频(最高最低价)
maxwindow = 120
minwindow = 15
industype = "Lv1"
knnthreshold = 0.75
newindus = getnewindus(maxwindow, minwindow, industype, knnthreshold)
