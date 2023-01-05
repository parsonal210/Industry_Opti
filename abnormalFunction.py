import pandas as pd
import scipy.io as scio
import numpy as np
from numba import jit


@jit(nopython=True)
def calr2(X, Y):
    return np.square(np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)),
                                      np.dot(X.T, Y)))-Y.mean()).sum()/np.square(Y-Y.mean()).sum()


def calinduspct(alldata, industype):
    alldata = pd.merge(alldata.groupby([industype+"code", "date"]).agg({"floatmarketcap": sum}).reset_index().
                     rename({"floatmarketcap": "floatsum"}, axis=1), alldata)  # 已检查，计算总市值
    alldata[industype+"share"] = alldata["floatmarketcap"]/alldata["floatsum"]*alldata["pctchg"]
    alldata = alldata.groupby([industype + "code", "date"]).agg({industype + "share": sum}).reset_index()
    return alldata


def calnpindus(indusdata, template, industype):
    nppct = dict()
    npdate = dict()
    for i in indusdata[industype+"code"].unique():
        tmpdata = indusdata[np.in1d(indusdata[industype+"code"], i)].reset_index(drop=True)
        tmpdata = pd.merge(template, tmpdata, how="left")
        nppct[i] = tmpdata[industype+"share"].values
        npdate[i] = tmpdata["date"].values
    return nppct, npdate


def calcorr(alldata, corr_thre=0.25):
    swnamecorr = list()
    zxnamecorr = list()
    corr = list()
    # 获取中信申万行业对应关系
    for i in alldata["zxcode"].unique():
        tmpdata = alldata[alldata["zxcode"] == i]
        zxnamecorr.append([i, pd.DataFrame(tmpdata["zxname"].value_counts()).reset_index().iloc[0]["index"]])
        tmpdata = pd.DataFrame(tmpdata["swcode"].value_counts()).reset_index()
        tmpdata["swcode"] = tmpdata["swcode"]/tmpdata["swcode"].sum()
        tmpdata = tmpdata[tmpdata["swcode"] >= corr_thre]["index"].to_list()
        # 若占比大于corr_thre，则认定行业对应关系
        for j in tmpdata:
            corr.append([i, j])
    for i in alldata["swcode"].unique():
        tmpdata = alldata[alldata["swcode"] == i]
        swnamecorr.append([i, pd.DataFrame(tmpdata["swname"].value_counts()).reset_index().iloc[0]["index"]])
        tmpdata = pd.DataFrame(tmpdata["zxcode"].value_counts()).reset_index()
        tmpdata["zxcode"] = tmpdata["zxcode"]/tmpdata["zxcode"].sum()
        tmpdata = tmpdata[tmpdata["zxcode"] >= corr_thre]["index"].to_list()
        for j in tmpdata:
            corr.append([j, i])
    swnamecorr = pd.DataFrame(swnamecorr).rename({0: "swcode", 1: "swname"}, axis=1)
    zxnamecorr = pd.DataFrame(zxnamecorr).rename({0: "zxcode", 1: "zxname"}, axis=1)
    corr = pd.DataFrame(corr).rename({0: "zxcode", 1: "swcode"}, axis=1).drop_duplicates()
    corr = pd.merge(pd.merge(corr, swnamecorr), zxnamecorr)
    return corr


def calpreknnindus(alldata, corrdate, maxwindow=120,minwindow=15):
    zxpct = calinduspct(alldata, "zx")
    swpct = calinduspct(alldata, "sw")
    template = pd.DataFrame(sorted(alldata["date"].unique())).rename({0: "date"}, axis=1)
    npzxpct, npzxdate = calnpindus(zxpct, template, "zx")
    npswpct, npswdate = calnpindus(swpct, template, "sw")
    # 获取中信申万行业指数数据
    alldata["const"] = 1
    alldata = alldata.drop(["floatmarketcap"], axis=1)
    rawdata = alldata.copy()
    outdata = dict()
    zxnto1out=pd.DataFrame()
    zx1to1out=pd.DataFrame()
    swnto1out=pd.DataFrame()
    sw1to1out=pd.DataFrame()
    for t in list(corrdate.keys()):  # 使用一段时期的数据匹配中信和申万行业
        alldata = rawdata[(rawdata["date"] > corrdate[t][0]) & (rawdata["date"] <= corrdate[t][1])]
        corr = calcorr(alldata)
        # 每个申万行业对应的中信行业信息
        tmp = corr.groupby("swcode").agg({"zxcode": "count"}).reset_index().rename({"zxcode": "count"}, axis=1)
        swnto1 = corr[corr["swcode"].isin(tmp[tmp["count"] != 1]["swcode"])].reset_index(drop=True)  # 每个申万行业有不止一个中信行业与之对应
        sw1to1 = corr[corr["swcode"].isin(tmp[tmp["count"] == 1]["swcode"])].reset_index(drop=True)  # 每个申万行业有唯一一个中信行业与之对应
        # 每个中信行业对应的申万行业信息
        tmp = corr.groupby("zxcode").agg({"swcode": "count"}).reset_index().rename({"swcode": "count"}, axis=1)
        zxnto1 = corr[corr["zxcode"].isin(tmp[tmp["count"] != 1]["zxcode"])]  # 每个中信行业有不止一个申万行业与之对应
        zx1to1 = corr[corr["zxcode"].isin(tmp[tmp["count"] == 1]["zxcode"])]  # 每个中信行业有唯一一个申万行业与之对应

        zxnto1 = zxnto1[~(zxnto1["swcode"].isin(swnto1["swcode"]))].reset_index(drop=True)  # 剔除swnto1已经包含的信息
        tmp = pd.DataFrame(zxnto1["zxcode"].value_counts()).reset_index()  # 每个中信行业对应的申万行业个数
        zx1to1 = pd.concat([zx1to1, zxnto1[zxnto1["zxcode"].isin(tmp[tmp["zxcode"] == 1]["index"])]]).reset_index(drop=True)
        zxnto1 = zxnto1[zxnto1["zxcode"].isin(tmp[tmp["zxcode"] != 1]["index"])].reset_index(drop=True)
        tmp = pd.concat([sw1to1, zx1to1]).drop_duplicates()
        corr1_1 = tmp[~(tmp["zxcode"].isin(zxnto1["zxcode"]))]
        zxnto1["t"]=t
        zx1to1["t"]=t
        swnto1["t"]=t
        sw1to1["t"]=t
        zxnto1out=pd.concat([zxnto1out,zxnto1])
        zx1to1out=pd.concat([zx1to1out,zx1to1])
        swnto1out=pd.concat([swnto1out,swnto1])
        sw1to1out=pd.concat([sw1to1out,sw1to1])
        swnto1dict = swnto1[["zxcode", "swcode"]].groupby("swcode").agg(list)["zxcode"].to_dict()
        zxnto1dict = zxnto1[["zxcode", "swcode"]].groupby("zxcode").agg(list)["swcode"].to_dict()
        same = pd.merge(alldata.drop("swname", axis=1), corr, how="outer")
        # same为中信申万一致的个股

        diff = same[same["swname"].isna()]
        same = same.dropna()
        tmpsw = same[same["swcode"].isin(swnto1["swcode"])]
        tmpsw = tmpsw[["date", "zxcode", "code", "zxname"]].rename({"zxname": "indusname", "zxcode": "induscode"},
                                                                   axis=1)
        tmpzx = same[same["zxcode"].isin(zxnto1["zxcode"])]
        tmpzx = tmpzx[["date", "swcode", "code", "swname"]].rename({"swname": "indusname", "swcode": "induscode"},
                                                                   axis=1)
        tmp = pd.merge(tmpsw[["date", "code", "induscode"]], tmpzx[["date", "code"]], how="right")
        tmpzx = pd.merge(tmp[tmp["induscode"].isna()][["date", "code"]], tmpzx)
        # 若中信或申万行业分类更详细，则保存为更详细的行业

        same = same[(~same["swcode"].isin(swnto1["swcode"])) & (~same["zxcode"].isin(zxnto1["zxcode"]))]
        same = same[["date", "zxcode", "code", "zxname"]].rename({"zxname": "indusname", "zxcode": "induscode"}, axis=1)
        # 否则以中信行业为基准

        finalsame = pd.concat([same, tmpzx, tmpsw])
        diff = pd.merge(corr[["zxcode", "zxname"]], diff.drop("zxname", axis=1))
        diff = pd.merge(corr[["swcode", "swname"]], diff.drop("swname", axis=1)).drop_duplicates()
        datelist=template["date"].to_list()
        olddata = rawdata[(rawdata["date"] >= datelist[datelist.index(corrdate[t][0])-maxwindow]) & 
                          (rawdata["date"] <= corrdate[t][1])] 
        olddata = olddata[olddata["code"].isin(diff["code"].unique())].sort_values(["code", "date"])
        # 获取个股在行业变动window天前的收益率，避免前window天缺失
        diffoutdata = list()
        for i in diff["code"].unique():
            tmpdata = diff[np.in1d(diff["code"], i)]
            #获取某个股的所有中信申万不对应记录
            tmpallpct = olddata[np.in1d(olddata["code"], i)].reset_index(drop=True)
            tmpallpct_date = tmpallpct["date"].values
            tmpallpct = tmpallpct[["pctchg", "const"]].values
            tmpzxlist = tmpdata["zxcode"].values
            tmpswlist = tmpdata["swcode"].values
            nptmpdate = tmpdata["date"].values
            for j in range(len(tmpdata)):
                #对每一条记录都单独处理，进行回归
                tmpswcode = tmpswlist[j]
                tmpzxcode = tmpzxlist[j]
                tmpdate = nptmpdate[j]
                enddate = int(np.where(tmpallpct_date == tmpdate)[0])
                if enddate < minwindow:
                    continue
                if enddate >= maxwindow:
                    window=maxwindow
                else:
                    window=enddate
                startdate = enddate-window+1
                #enddate即当前一条记录的日期，回归的最后一天
                #此if判断新上市不满window天的情况
                tmpdatelist = tmpallpct_date[startdate:(enddate+1)]
                #tmpdatelist为当前一天到前window天的所有date
                tmppct = tmpallpct[startdate:(enddate+1)]
                try:
                    zxstartindex = np.where(npzxdate[tmpzxcode] == tmpdatelist[0])[0][0]
                    zxendindex = np.where(npzxdate[tmpzxcode] == tmpdatelist[-1])[0][0]
                    #获取中信行业中，个股window天的最前和最后一天index
                    #由于中信和申万行业调整和行业新设立
                    #可能导致window天的最前一天在中信中没有index进而报错，使用try
                except:
                    continue
                if zxendindex-zxstartindex+1 == window:
                    tmpzx = npzxpct[tmpzxcode][zxstartindex:zxendindex+1]
                    #如果中信中获取的数据量和个股相同，则根据index得到中信pct
                else:
                    tmpzx = npzxpct[tmpzxcode][np.in1d(npzxdate[tmpzxcode], tmpdatelist)]
                    #如果个股存在停牌等情况，当前日期向前window的交易日信息可能不同，需要使用in1d排除掉这些交易日信息
                try:
                    swstartindex = np.where(npswdate[tmpswcode] == tmpdatelist[0])[0][0]
                    swendindex = np.where(npswdate[tmpswcode] == tmpdatelist[-1])[0][0]
                except:
                    continue
                if swendindex-swstartindex+1 == window:
                    tmpsw = npswpct[tmpswcode][swstartindex:swendindex+1]
                else:
                    tmpsw = npswpct[tmpswcode][np.in1d(npswdate[tmpswcode], tmpdatelist)]
                if len(tmpsw)+len(tmpzx)+len(tmppct) < 3*window:
                    continue
                    #若不满window天则跳过
                if calr2(tmppct, tmpzx) >= calr2(tmppct, tmpsw):
                    diffoutdata.append([i, tmpdate, "zx"])
                else:
                    diffoutdata.append([i, tmpdate, "sw"])
        # 对中信申万不一致个股以收益率R2进行归类
        diffoutdata=pd.DataFrame(diffoutdata)
        diffoutdata.columns=["code","date","indus"]
        tmpout=pd.merge(diff,diffoutdata)
        tmpzx=tmpout[tmpout["indus"]=="zx"].drop(["swcode","swname","pctchg","indus"],axis=1).rename({"zxcode":"induscode","zxname":"indusname"},axis=1)
        tmpsw=tmpout[tmpout["indus"]=="sw"].drop(["zxcode","zxname","pctchg","indus"],axis=1).rename({"swcode":"induscode","swname":"indusname"},axis=1)
        diffoutdata=pd.concat([tmpzx,tmpsw])
        tmpzxnto1=diffoutdata[diffoutdata["induscode"].isin(zxnto1["zxcode"])]
        tmpswnto1=diffoutdata[diffoutdata["induscode"].isin(swnto1["swcode"])]
        swnto1outdata=list()
        zxnto1outdata=list()
            # 将个股按照收益率R2区分到更详细的行业分类中
        for i in tmpswnto1["code"].unique():
            tmpdata=tmpswnto1[np.in1d(tmpswnto1["code"],i)]
            tmpallpct=olddata[np.in1d(olddata["code"],i)].reset_index(drop=True)
            tmpallpct_date=tmpallpct["date"].values
            nptmpallpct=tmpallpct[["pctchg","const"]].values
            tmpswlist=tmpdata["induscode"].values
            nptmpdate=tmpdata["date"].values
            for j in range(len(tmpdata)):
                tmpswcode=tmpswlist[j]
                tmpdate=nptmpdate[j] 
                enddate = int(np.where(tmpallpct_date == tmpdate)[0])
                if enddate < minwindow:
                    continue
                if enddate >= maxwindow:
                    window=maxwindow
                else:
                    window=enddate
                startdate = enddate-window+1
                tmpdatelist=tmpallpct_date[startdate:enddate+1]
                tmppct=nptmpallpct[startdate:enddate+1]
                tmpr2=list()
                for k in swnto1dict[tmpswcode]:
                    try:
                        zxstartindex=np.where(npzxdate[k]==tmpdatelist[0])[0][0]
                        zxendindex=np.where(npzxdate[k]==tmpdatelist[-1])[0][0]
                    except:
                        continue
                    if(zxendindex-zxstartindex+1==window):
                        tmpzx=npzxpct[k][zxstartindex:zxendindex+1]
                    else:
                        tmpzx=npzxpct[k][np.in1d(npzxdate[k],tmpdatelist)]                
                    if(len(tmpzx)+len(tmppct)<2*window):
                        continue
                    tmpr2.append([k,calr2(tmppct,tmpzx)])
                if(len(tmpzx)+len(tmppct)<2*window):
                    continue
                try:
                    tmpr2=np.array(tmpr2)
                    swnto1outdata.append([i,nptmpdate[j],tmpr2[np.where(tmpr2[:,1]==np.max(tmpr2[:,1]))[0][0]][0]])
                except:
                    continue
        # 将个股按照收益率R2区分到更详细的行业分类中
        for i in tmpzxnto1["code"].unique():
            tmpdata=tmpzxnto1[np.in1d(tmpzxnto1["code"],i)]
            tmpallpct=olddata[np.in1d(olddata["code"],i)].reset_index(drop=True)
            tmpallpct_date=tmpallpct["date"].values
            nptmpallpct=tmpallpct[["pctchg","const"]].values
            tmpzxlist=tmpdata["induscode"].values
            nptmpdate=tmpdata["date"].values
            for j in range(len(tmpdata)):
                tmpzxcode=tmpzxlist[j]
                tmpdate=nptmpdate[j]
                enddate = int(np.where(tmpallpct_date == tmpdate)[0])
                if enddate < minwindow:
                    continue
                if enddate >= maxwindow:
                    window=maxwindow
                else:
                    window=enddate
                startdate = enddate-window+1
                tmpdatelist=tmpallpct_date[startdate:enddate+1]
                tmppct=nptmpallpct[startdate:enddate+1]
                tmpr2=list()            
                for k in zxnto1dict[tmpzxcode]:
                    try:
                        swstartindex=np.where(npswdate[k]==tmpdatelist[0])[0][0]
                        swendindex=np.where(npswdate[k]==tmpdatelist[-1])[0][0]
                    except:
                        continue
                    if(swendindex-swstartindex+1==window):
                        tmpsw=npswpct[k][swstartindex:swendindex+1]
                    else:
                        tmpsw=npswpct[k][np.in1d(npswdate[k],tmpdatelist)]       
                    if(len(tmpsw)+len(tmppct)<2*window):
                        continue
                    tmpr2.append([k,calr2(tmppct,tmpsw)])
                if(len(tmpsw)+len(tmppct)<2*window):
                    continue
                try:
                    tmpr2=np.array(tmpr2)
                    zxnto1outdata.append([i,nptmpdate[j],tmpr2[np.where(tmpr2[:,1]==np.max(tmpr2[:,1]))[0][0]][0]])
                except:
                    continue
        try:
            zxnto1outdata=pd.DataFrame(zxnto1outdata)
            zxnto1outdata.columns=["code","date","induscode"]
            zxnto1outdata=pd.merge(corr[["swcode","swname"]].rename({"swcode":"induscode","swname":"indusname"},axis=1).drop_duplicates(),zxnto1outdata)
        except:
            pass
        try:
            swnto1outdata=pd.DataFrame(swnto1outdata)
            swnto1outdata.columns=["code","date","induscode"]
            swnto1outdata=pd.merge(corr[["zxcode","zxname"]].rename({"zxcode":"induscode","zxname":"indusname"},axis=1).drop_duplicates(),swnto1outdata)
        except:
            pass
        nto1outdata=pd.concat([zxnto1outdata,swnto1outdata])
        diffoutdata=diffoutdata[(~diffoutdata["induscode"].isin(swnto1["swcode"]))&(~diffoutdata["induscode"].isin(zxnto1["zxcode"]))]
        diffoutdata=pd.concat([diffoutdata,nto1outdata])
        tmpoutdata=pd.concat([finalsame,diffoutdata]).drop("const",axis=1)
        tmpoutdata=pd.merge(tmpoutdata,corr1_1.rename({"swcode":"induscode"},axis=1),how="left")
        # 处理中信多个行业对应申万多个行业的情况，避免输出行业中同一个indusname有多个induscode
        tmpoutdata["zxcode"]=tmpoutdata["zxcode"].fillna(tmpoutdata["induscode"])
        tmpoutdata=tmpoutdata.drop("induscode",axis=1).rename({"zxcode":"induscode"},axis=1)
        tmpdata=tmpoutdata[["indusname","induscode"]].drop_duplicates()
        tmpname=list()
        for i in tmpdata["induscode"].unique():
            tmp=tmpdata[tmpdata["induscode"]==i]
            tmpname.append([i,pd.DataFrame(tmp["indusname"].value_counts()).reset_index().iloc[0]["index"]])
        tmpoutdata=pd.merge(tmpoutdata.drop("indusname",axis=1),pd.DataFrame(tmpname).rename({0:"induscode",1:"indusname"},axis=1))
        outdata[t]=tmpoutdata
    finaldata=pd.DataFrame()
    for i in outdata.keys():
        finaldata=pd.concat([finaldata,outdata[i]])
    finaldata=finaldata.drop(["swname","zxname"],axis=1)
    zxnto1out.to_excel("output/zxnto1.xlsx")
    zx1to1out.to_excel("output/zx1to1.xlsx")
    swnto1out.to_excel("output/swnto1.xlsx")
    sw1to1out.to_excel("output/sw1to1.xlsx")
    return finaldata


def caldiffr2(diff,alldata,maxwindow=120,minwindow=15):
    output=dict()
    zxpct = calinduspct(alldata, "zx")
    swpct = calinduspct(alldata, "sw")
    template = pd.DataFrame(sorted(alldata["date"].unique())).rename({0: "date"}, axis=1)
    npzxpct, npzxdate = calnpindus(zxpct, template, "zx")
    npswpct, npswdate = calnpindus(swpct, template, "sw")
    for i in sorted(diff["code"].unique()):
        tmpdata = diff[np.in1d(diff["code"], i)]
        tmpallpct = alldata[np.in1d(alldata["code"], i)].reset_index(drop=True)
        tmpallpct_date = tmpallpct["date"].values
        tmpallpct = tmpallpct[["pctchg", "const"]].values
        tmpzxlist = tmpdata["zxcode"].values
        tmpswlist = tmpdata["swcode"].values
        nptmpdate = tmpdata["date"].values
        output[i]=pd.DataFrame()
        for j in range(len(tmpdata)):
            #对每一条记录都单独处理，进行回归
            tmpdf=tmpdata.iloc[j]
            tmpswcode = tmpswlist[j]
            tmpzxcode = tmpzxlist[j]
            tmpdate = nptmpdate[j]
            enddate = int(np.where(tmpallpct_date == tmpdate)[0])
            if enddate < minwindow:
                continue
            if enddate >= maxwindow:
                window=maxwindow
            else:
                window=enddate
            startdate = enddate-window+1
            #此if判断新上市不满window天的情况
            tmpdatelist = tmpallpct_date[startdate:(enddate+1)]
            #tmpdatelist为当前一天到前window天的所有date
            tmppct = tmpallpct[startdate:(enddate+1)]
            try:
                zxstartindex = np.where(npzxdate[tmpzxcode] == tmpdatelist[0])[0][0]
                zxendindex = np.where(npzxdate[tmpzxcode] == tmpdatelist[-1])[0][0]
                #获取中信行业中，个股window天的最前和最后一天index
                #由于中信和申万行业调整和行业新设立
                #可能导致window天的最前一天在中信中没有index进而报错，使用try
            except:
                continue
            if zxendindex-zxstartindex+1 == window:
                tmpzx = npzxpct[tmpzxcode][zxstartindex:zxendindex+1]
                #如果中信中获取的数据量和个股相同，则根据index得到中信pct
            else:
                tmpzx = npzxpct[tmpzxcode][np.in1d(npzxdate[tmpzxcode], tmpdatelist)]
                #如果个股存在停牌等情况，当前日期向前window的交易日信息可能不同，需要使用in1d排除掉这些交易日信息
            try:
                swstartindex = np.where(npswdate[tmpswcode] == tmpdatelist[0])[0][0]
                swendindex = np.where(npswdate[tmpswcode] == tmpdatelist[-1])[0][0]
            except:
                continue
            if swendindex-swstartindex+1 == window:
                tmpsw = npswpct[tmpswcode][swstartindex:swendindex+1]
            else:
                tmpsw = npswpct[tmpswcode][np.in1d(npswdate[tmpswcode], tmpdatelist)]
            if len(tmpsw)+len(tmpzx)+len(tmppct) < 3*window:
                continue
                #若不满window天则跳过
            tmpdf=pd.DataFrame(tmpdf).T
            tmpdf["swR2"]=calr2(tmppct, tmpsw)
            tmpdf["zxR2"]=calr2(tmppct, tmpzx)
            output[i]=pd.concat([output[i],tmpdf])
    finaloutput=pd.DataFrame()
    for i in output.keys():
        finaloutput=pd.concat([finaloutput,output[i]])
    return finaloutput



def outputdiffr2(alldata,data,daydf):
    alldata["const"] = 1
    diffknn = pd.read_excel("output/diffknn.xlsx",index_col=0)
    diffknn = diffknn[["date_period","code","type"]]
    # getknndata获取KNN模型所需的数据
    diffknn = diffknn[diffknn["type"]=="R2"]
    quarterdate = sorted(data["date_period"].unique()) 
    lastknn = diffknn.copy()
    diffknn = pd.merge(diffknn,daydf)
    diff = pd.merge(diffknn,alldata)
    lastknn["lastdate_period"] = lastknn["date_period"].apply(lambda x:quarterdate[quarterdate.index(x)-1])
    lastknn = lastknn.rename({"lastdate_period":"date_period","date_period":"next_period"},axis=1)
    lastknn = pd.merge(lastknn,daydf)
    last = pd.merge(lastknn,alldata)
    diffout = caldiffr2(diff,alldata)
    lastout = caldiffr2(last,alldata)
    diffout = diffout.groupby(["date_period","code"]).agg({"swR2":"mean","zxR2":"mean"}).reset_index()
    lastout = lastout.groupby(["next_period","code"]).agg({"swR2":"mean","zxR2":"mean"}).reset_index()
    lastout = lastout.rename({"swR2":"lastswR2","zxR2":"lastzxR2","next_period":"date_period"},axis=1)
    output = pd.merge(diffout,lastout,how="outer")
    diffknn = pd.read_excel("output/diffknn.xlsx",index_col=0)
    diffknn = pd.merge(diffknn,output,how="left")
    diffknn.to_excel("output/diffknn.xlsx")