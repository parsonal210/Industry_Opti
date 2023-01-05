import scipy.io as scio
import pandas as pd
import numpy as np
import os

def caldaydf(alldata,quarterdate):
    dailydata = alldata[alldata["date"] >= 20130601][["code", "date"]].reset_index(drop=True)
    daylist = sorted(dailydata["date"].unique())
    daydf = pd.DataFrame()
    for i in range(len(quarterdate)):
        tmp = pd.DataFrame(daylist[:daylist.index(quarterdate[i])+1]).rename({0:"date"},axis=1)
        tmp["date_period"] = quarterdate[i]
        daydf = pd.concat([daydf,tmp])
        daylist = daylist[daylist.index(quarterdate[i])+1:]
    return daydf

def indusoutput(indusdata, industype):
    codedata = scio.loadmat("data/code.mat")
    datedata = scio.loadmat("data/date.mat")
    data = dict()
    data["code"] = np.repeat(codedata["code"], datedata["date"].shape[0])
    data["date"] = np.tile(datedata["date"], (codedata["code"].shape[0], 1)).flatten()
    data = pd.DataFrame(data)
    data = data[data["date"] >= 20140401]
    indusname = indusdata[["induscode", "indusname"]].drop_duplicates()
    indusname["number"] = indusname["induscode"]
    indusname = indusname.rename({"induscode": "code", "indusname": "name"}, axis=1)
    indusname = indusname.groupby("code").agg({"name": max, "number": max}).reset_index()
    indusname.to_excel("data/constant/new"+industype+"Info.xlsx")
    indusdata = pd.merge(data, indusdata, how="left")
    indusdata = np.array(indusdata["induscode"]).reshape(len(indusdata["code"].unique()),
                                                         len(indusdata["date"].unique()))
    scio.savemat("data/riskFactor/new"+industype+".mat", {"new"+industype: indusdata})


def calindusprice(industype="newLv1", calmindata=False):
    if industype == "Lv1":
        nowindus = "new"+industype
    elif industype == "Lv2":
        nowindus = "new"+industype
    else:
        nowindus = industype
    startdate=20130601
    enddate=99999999
    codeFile = "data/code.mat"
    dateFile = "data/date.mat"
    codedata = scio.loadmat(codeFile)
    datedata = scio.loadmat(dateFile)
    data = dict()
    data["code"] = np.repeat(codedata["code"], datedata["date"].shape[0])
    data["date"] = np.tile(datedata["date"], (codedata["code"].shape[0], 1)).flatten()
    files = dict()
    files["ipo"] = "data/stockSelect/ipo.mat"
    for i in files.keys():
        data[i] = scio.loadmat(files[i])[i].flatten()
    alldata = pd.DataFrame.from_dict(data)
    codeFile = "data/code.mat"
    dateFile = "data/date.mat"
    codedata = scio.loadmat(codeFile)
    datedata = scio.loadmat(dateFile)
    data = dict()
    startindex = datedata["date"].flatten().tolist().index(20140401)
    datedata["date"] = datedata["date"][datedata["date"].flatten().tolist().index(20140401):]
    data["code"] = np.repeat(codedata["code"], datedata["date"].shape[0])
    data["date"] = np.tile(datedata["date"],(codedata["code"].shape[0],1)).flatten()
    # 根据每只个股的数据时间转为一维arraycalindusprice(industype, calmindata=False)
    files["suspend"] = "data/stockSelect/suspend.mat"
    files["floatshare"] = "data/share/floatshare.mat"
    files["abnormal"] = "data/stockSelect/abnormal.mat"
    files["ipo"] = "data/stockSelect/ipo.mat"
    files["listed"] = "data/stockSelect/listed.mat"
    files["open"] = "data/price/open.mat"
    files["close"] = "data/price/close.mat"
    files["amount"] = "data/price/amount.mat"
    for i in files.keys():
        data[i] = scio.loadmat(files[i])[i][:, startindex:].flatten()
    if(nowindus[:3]=="new"):      
        files[nowindus] = "data/riskFactor/"+nowindus+".mat"
        data[nowindus] = scio.loadmat(files[nowindus])[nowindus].flatten()
    alldata=pd.DataFrame.from_dict(data)
    alldata=alldata[alldata["date"]>=20140401]
    alldata["open_pro"]=alldata["floatshare"]*alldata["open"]
    alldata["close_pro"]=alldata["floatshare"]*alldata["close"]
    abnormal=alldata.groupby("code").apply(lambda x:x["abnormal"]-x["abnormal"].shift()).reset_index().rename({"level_1":"index","abnormal":"normal"},axis=1)
    abnormal["normal"] = abnormal["normal"].fillna(0)
    alldata=pd.merge(abnormal, alldata.reset_index()).drop("index",axis=1)
    # 考虑到申万行业变动，只处理2014年以来数据
    if(nowindus[:3]!="new"):      
        zxindustry = pd.read_excel("data/riskFactor/industryZX.xls")[["code", "ann_dt", "del_dt", industype[-3:]+"code",
                                                                      industype[-3:]+"name", "standard"]]
        zxindustry = zxindustry[zxindustry["standard"] == 37].copy()  # 只保留中信行业2019分类，检查过了没什么问题
        zxindustry["del_dt"] = np.where(zxindustry["del_dt"] == 0, 99999999, zxindustry["del_dt"])
        zxindustry = zxindustry[zxindustry["del_dt"] > startdate]
        # 将中信行业分类信息和基础数据合并
        alldata = pd.merge(zxindustry, alldata)  # 已检查，set(alldata['code']).difference(set(zxindustry['code']))是空的
        alldata = alldata[alldata["del_dt"] > alldata["date"]]  # 已检查，要求成分股剔除日在当天交易日以后
        alldata = alldata[alldata["ann_dt"] <= alldata["date"]]  # 新增这一行，避免使用刚上市股票的行业分类，同时要提前处理股票代码变更
        alldata = alldata.drop(["ann_dt", "del_dt", "standard"], axis=1)
        alldata = alldata.rename({industype[-3:]+"code": nowindus, industype[-3:]+"name": "zxname"}, axis=1)
        # 处理得到中信分类
    rawdata=alldata.copy()
    alldata=alldata[(alldata["abnormal"]==1)&(alldata["ipo"]==1)&(alldata["listed"]==1)].dropna()
    alldata=alldata.drop(["abnormal","ipo","listed"],axis=1)
    #去掉异常情况个股
    npdate=scio.loadmat(dateFile)["date"]
    npinduscode=alldata[nowindus].drop_duplicates().sort_values().values
    induscode=np.repeat(npinduscode,npdate.shape[0])
    indusdate=np.tile(npdate,(npinduscode.shape[0],1)).flatten()
    template=pd.DataFrame([induscode,indusdate]).T.rename({0:nowindus,1:"date"},axis=1)
    datedata=scio.loadmat(dateFile)["date"].flatten().tolist()
    datedata.extend(90*[99999999,])
    alldate=sorted(alldata["date"].unique())
    lastshare=alldata.groupby("code").apply(lambda x:x["floatshare"].shift()).reset_index()
    lastshare=lastshare.rename({"floatshare":"lastshare"},axis=1)
    alldata=pd.merge(alldata.reset_index().rename({"index":"level_1"},axis=1),lastshare).drop("level_1",axis=1)
    alldata["plus"]=(alldata["floatshare"]==alldata["lastshare"]).map({False:0,True:1})
    #判断增发
    tmpdata=alldata.groupby("code").agg({"date":min}).reset_index()
    tmpdata["tmp"]=1
    tmpdata=pd.merge(alldata,tmpdata,how="left")
    alldata=tmpdata[tmpdata["tmp"].isna()].drop("tmp",axis=1)
    tmpdata=tmpdata[tmpdata["tmp"]==1].drop("tmp",axis=1)
    tmpdata["plus"]=1
    alldata=pd.concat([tmpdata,alldata]).drop("lastshare",axis=1)
    alldate=sorted(alldata["date"].unique())
    #避免第一天没有lastshare导致被判断为增发
    daydata=dict()
    for t in range(len(alldate)):
        daydata[t]=rawdata[rawdata["date"]==alldate[t]]
    amountdata=pd.merge(template,alldata.groupby([nowindus,"date"]).agg({"amount":sum}).reset_index(),how="left")
    amountdata=amountdata["amount"].values.reshape(npinduscode.shape[0],npdate.shape[0])
    scio.savemat("data/indexprice/"+nowindus+"/amount_industry.mat",{"amount_industry":amountdata})
    opendict=dict()
    closedict=dict()
    for i in alldata[nowindus].unique():
        indusdata=alldata[alldata[nowindus]==i]
        indusdata=indusdata[indusdata["normal"]==0]
        indusdata=indusdata[indusdata["plus"]==1]
        indusdate=sorted(indusdata["date"].unique())
        opendict[i]=[[i,indusdate[0],100],]
        tmpdata=indusdata[indusdata["date"]==indusdate[0]]
        closedict[i]=[[i,indusdate[0],100*tmpdata["close_pro"].sum()/tmpdata["open_pro"].sum()],]
        for t in range(1,len(indusdate)):
            tmpdata=indusdata[indusdata["date"]==indusdate[t]]
            lastdata=daydata[alldate.index(indusdate[t])-1]
            lastopen=lastdata["open_pro"][lastdata["code"].isin(tmpdata["code"])].sum()
            nowopen=tmpdata["open_pro"].sum()
            opendict[i].append([i,indusdate[t],opendict[i][-1][-1]*nowopen/lastopen])
            closedict[i].append([i,indusdate[t],opendict[i][-1][-1]*tmpdata["close_pro"].sum()/tmpdata["open_pro"].sum()])
    opendata=pd.DataFrame()
    closedata=pd.DataFrame()
    for i in opendict.keys():
        opendict[i]=pd.DataFrame(opendict[i]).rename({0:nowindus,1:"date",2:"open_industry"},axis=1)
        opendata=pd.concat([opendata,opendict[i]])
        closedict[i]=pd.DataFrame(closedict[i]).rename({0:nowindus,1:"date",2:"close_industry"},axis=1)
        closedata=pd.concat([closedata,closedict[i]])
    opendata=pd.merge(opendata,template,how="right")
    closedata=pd.merge(closedata,template,how="right")
    opendata=opendata["open_industry"].values.reshape(npinduscode.shape[0],npdate.shape[0])
    closedata=closedata["close_industry"].values.reshape(npinduscode.shape[0],npdate.shape[0])
    scio.savemat("data/indexprice/"+nowindus+"/open_industry.mat",{"open_industry":opendata})
    scio.savemat("data/indexprice/"+nowindus+"/close_industry.mat",{"close_industry":closedata})
    scio.savemat("data/indexprice/"+nowindus+"/returnDay_industry.mat",{"returnDay_industry":closedata/opendata-1})
    if (calmindata==True):
        codedir="E:/min分钟频/code/"
        lowdir="E:/min分钟频/low/"
        highdir="E:/min分钟频/high/"
        high_low=pd.DataFrame()
        for i in os.listdir(codedir):
            if(int(i[:-4])<20140401):
                continue
            tmpdata=alldata[alldata["date"]==int(i[:-4])]
            code=scio.loadmat(codedir+i)["code"]
            low=scio.loadmat(lowdir+i)["low"]
            high=scio.loadmat(highdir+i)["high"]
            time=np.tile(list(range(1,241)),code.shape[0])
            code=np.repeat(code,low.shape[1])
            low=low.flatten()
            high=high.flatten()
            mindata=pd.DataFrame(code).rename({0:"code"},axis=1)
            mindata["low"]=low
            mindata["high"]=high
            mindata["time"]=time
            mindata=pd.merge(mindata,tmpdata)
            mindata["low"]=mindata["low"]*mindata["floatshare"]
            mindata["high"]=mindata["high"]*mindata["floatshare"]
            tmpdata=mindata[mindata["time"]==1][["code","open_pro","newLv1","newLv2"]]
            open_pro=tmpdata.groupby(nowindus).agg({"open_pro":sum}).reset_index()
            mindata=mindata.groupby([nowindus,"time"]).agg({"low":sum,"high":sum}).reset_index()
            mindata=mindata.groupby(nowindus).agg({"low":min,"high":max}).reset_index()
            mindata=pd.merge(mindata,open_pro)
            mindata["low"]=mindata["low"]/mindata["open_pro"]
            mindata["high"]=mindata["high"]/mindata["open_pro"]
            mindata=mindata.drop("open_pro",axis=1)
            mindata["date"]=int(i[:-4])
            high_low=pd.concat([high_low,mindata])
        high_low=pd.merge(template,high_low,how="left")
        highdata=high_low["high"].values.reshape(npinduscode.shape[0],npdate.shape[0])
        lowdata=high_low["low"].values.reshape(npinduscode.shape[0],npdate.shape[0])
        highdata=opendata*highdata
        lowdata=opendata*lowdata
        scio.savemat("data/indexprice/"+nowindus+"/high_industry.mat",{"high_industry":highdata})
        scio.savemat("data/indexprice/"+nowindus+"/low_industry.mat",{"low_industry":lowdata})