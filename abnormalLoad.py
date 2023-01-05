import pandas as pd
import scipy.io as scio
import numpy as np
import bottleneck as bn
import os


def winsorize(x1):
    with np.errstate(invalid='ignore'):
        median_x1 = bn.nanmedian(x1, axis=0)
        median_x11 = bn.nanmedian(np.abs(x1 - median_x1), axis=0)
        up_limit = median_x1 + 5 * median_x11
        down_limit = median_x1 - 5 * median_x11
        x1_winsorize = (x1 > up_limit) * up_limit + (x1 < down_limit) * down_limit + \
                       (x1 <= up_limit) * (x1 >= down_limit) * x1
        return x1_winsorize



def getdata(industype="Lv1", startdate=20110601, enddate=99999999):
    codeFile = "data/code.mat"
    dateFile = "data/date.mat"
    codedata = scio.loadmat(codeFile)
    datedata = scio.loadmat(dateFile)
    data = dict()
    data["code"] = np.repeat(codedata["code"], datedata["date"].shape[0])
    data["date"] = np.tile(datedata["date"], (codedata["code"].shape[0], 1)).flatten()
    # 根据每只个股的数据时间转为一维array
    files = dict()
    files["returnDay"] = "data/price/returnDay.mat"
    files["floatmarketcap"] = "data/marketvalue/floatmarketcap.mat"  # 已检查，把floatshare(流通股数)改成floatmarketcap(流通市值，万元)
    files["abnormal"] = "data/stockSelect/abnormal.mat"
    files["ipo"] = "data/stockSelect/ipo.mat"
    files["suspend"] = "data/stockSelect/suspend.mat"
    files["listed"] = "data/stockSelect/listed.mat"
    for i in files.keys():
        data[i] = scio.loadmat(files[i])[i].flatten()
    alldata = pd.DataFrame.from_dict(data)
    alldata = alldata[alldata["date"] >= startdate]
    alldata = alldata[alldata["date"] <= enddate]
    alldata = alldata.rename({"returnDay": "pctchg"}, axis=1)
    alldata = alldata[(alldata["abnormal"] == 1) & (alldata["ipo"] == 1) & (alldata["suspend"] == 1) &
                      (alldata["listed"] == 1)].dropna()
    # 去掉异常情况个股
    zxindustry = pd.read_excel("data/riskFactor/industryZX.xls")[["code", "ann_dt", "del_dt", industype+"code",
                                                                  industype+"name", "standard"]]
    zxindustry = zxindustry[zxindustry["standard"] == 37].copy()  # 只保留中信行业2019分类，检查过了没什么问题
    zxindustry["del_dt"] = np.where(zxindustry["del_dt"] == 0, 99999999, zxindustry["del_dt"])
    zxindustry = zxindustry[zxindustry["del_dt"] > startdate]

    # 将中信行业分类信息和基础数据合并
    alldata = pd.merge(zxindustry, alldata)  # 已检查，set(alldata['code']).difference(set(zxindustry['code']))是空的
    alldata = alldata[alldata["del_dt"] > alldata["date"]]  # 已检查，要求成分股剔除日在当天交易日以后
    alldata = alldata[alldata["ann_dt"] <= alldata["date"]]  # 新增这一行，避免使用刚上市股票的行业分类，同时要提前处理股票代码变更
    # alldata = pd.merge(alldata, alldata.groupby(["code", "date"]).agg({"del_dt": "min"}).reset_index())  # 已检查，可删除
    alldata = alldata.drop(["ann_dt", "del_dt", "abnormal", "ipo", "suspend", "listed", "standard"], axis=1)
    alldata = alldata.rename({industype+"code": "zxcode", industype+"name": "zxname"}, axis=1)
    swindustry = pd.read_excel("data/riskFactor/industrySW.xls")[["code", "ann_dt", "del_dt", industype+"code",
                                                                  industype+"name"]]
    swindustry["del_dt"] = np.where(swindustry["del_dt"] == 0, 99999999, swindustry["del_dt"])
    swindustry = swindustry[swindustry["del_dt"] > startdate]

    # 以下处理为inner，即要求中信和申万在给定交易日必须都给出了行业分类，如果有一家没有给行业分类就是nan
    alldata = pd.merge(swindustry, alldata)  # 已检查，set(alldata['code']).difference(set(swindustry['code']))是空的
    alldata = alldata[alldata["del_dt"] > alldata["date"]]  # 已检查，要求成分股剔除日在当天交易日以后
    alldata = alldata[alldata["ann_dt"] <= alldata["date"]]  # 新增这一行，避免使用刚上市股票的行业分类，同时要提前处理股票代码变更
    # alldata = pd.merge(alldata, alldata.groupby(["code", "date"]).agg({"del_dt": "min"}).reset_index())  # 已检查，可删除
    alldata = alldata.drop(["ann_dt", "del_dt"], axis=1).rename({industype+"code": "swcode",
                                                                 industype+"name": "swname"}, axis=1)
    # 处理得到中信申万分类
    return alldata


def getknndata():
    rootdir = "data/financial_period/"  # 已检查，原先要保证文件夹里没有其他数据，现在不用
    filename = ['code', 'date_period', 'asset', 'equity', 'invested_capital', 'net_profit', 'revenue']
    basicdata = dict()
    data = dict()
    for i in filename:
        basicdata[i] = scio.loadmat(rootdir+i+'.mat')[i]
    data["date_period"] = np.tile(basicdata["date_period"], (basicdata["code"].shape[0], 1)).flatten()
    data["code"] = np.repeat(basicdata["code"], basicdata["date_period"].shape[0])
    del basicdata["code"]
    del basicdata["date_period"]
    for i in basicdata.keys():
        data[i] = basicdata[i].flatten()
    del basicdata
    data = pd.DataFrame.from_dict(data)
    data["year"] = (data["date_period"]/10000).astype(int)
    data["month"] = data["date_period"].astype(str).str[4:6].astype(int)
    quarterdate = sorted(data["date_period"].unique())
    data = data.drop_duplicates()
    data = data[data["year"]>=2013]
    tmpdata = data.groupby(["code"]).apply(lambda x: x[["net_profit", "revenue"]]-x[["net_profit", "revenue"]]
                                           .shift(1)).reset_index()
    tmpdata = pd.merge(data.reset_index().drop(["net_profit", "revenue"], axis=1), tmpdata)
    data = pd.concat([tmpdata[tmpdata["month"] != 4], data[data["month"] == 4]])  # 已检查，得到单季度财务数据
    data = data.drop("index", axis=1).dropna().sort_values(["code", "date_period"]).reset_index(drop=True)

    # 计算财务指标，先去极值然后sigmoid
    data["assetturnover"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: x["revenue"]/((x["asset"].shift(1)+x["asset"])/2)).reset_index()[0])))
    data["profitmargin"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: x["net_profit"]/x["revenue"]).reset_index()[0])))
    data["debt2AssetRatio"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: (x["asset"]-x["equity"])/x["asset"]).reset_index()[0])))
    data["ROA"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: x["net_profit"]/((x["asset"].shift(1)+x["asset"])/2)).reset_index()[0])))
    data["ROE"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: x["net_profit"]/((x["equity"].shift(1)+x["equity"])/2)).reset_index()[0])))
    data["ROIC"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: x["net_profit"]/((x["invested_capital"].shift(1) + x["invested_capital"])/2)).reset_index()[0])))
    data["revenue_MA_growth"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: (x["revenue"]+x["revenue"].shift(1)+x["revenue"].shift(2)+x["revenue"].shift(3))/
               (x["revenue"].shift(1)+x["revenue"].shift(2)+x["revenue"].shift(3)+x["revenue"].shift(4))).reset_index()["revenue"])))
    data["NP_MA_growth"] = 1/(1+np.exp(-winsorize(data.groupby("code").apply(
        lambda x: (x["net_profit"]+x["net_profit"].shift(1)+x["net_profit"].shift(2)+x["net_profit"].shift(3))/
               (x["net_profit"].shift(1)+x["net_profit"].shift(2)+x["net_profit"].shift(3)+x["net_profit"].shift(4))).reset_index()
                                                  ["net_profit"])))
    data = data.dropna().drop(["asset", "equity", "invested_capital", "net_profit", "revenue", "year", "month"], axis=1)

    # 以最后一个记录补充缺少的最新财务数据
    lastdata = pd.merge(data, data.groupby("code").agg({"date_period": max}).reset_index())
    newdata = pd.DataFrame()
    for i in range(len(lastdata)):
        tmpdata = lastdata.iloc[i].copy()
        quarindex = quarterdate.index(tmpdata["date_period"])
        for j in range(quarindex+1, len(quarterdate)):
            tmpdata["date_period"] = quarterdate[j]
            newdata = pd.concat([newdata, pd.DataFrame(tmpdata).T])
    data = pd.concat([newdata, data])
    return data


def getquarterdate(startdate = 20131031):
    quarterdate = sorted(pd.Series(scio.loadmat("data/financial_period/date_period.mat")["date_period"].flatten()).unique())
    quarterdate = quarterdate[quarterdate.index(20131031):]
    return quarterdate