from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from io import BytesIO
import gc
import requests

app = FastAPI()

# 定义输入模型
class AnalyzeInput(BaseModel):
    file_url: str

@app.post("/analyze/")
async def analyze(input: AnalyzeInput):
    #response = requests.get(input.file_url)
    response = requests.get(input.file_url, allow_redirects=True)
    if response.status_code != 200:
        return {"error": "文件下载失败"}
    df = pd.read_excel(BytesIO(response.content), engine="openpyxl")

    # 预处理与聚类逻辑（你的核心代码）
    df.columns = df.columns.str.lower()
    df['from_address'] = df['from_address'].astype('category')
    df['to_address'] = df['to_address'].astype('category')

    df = df.rename(columns={
        'timestamp':'ts_chain', 'received_timestamp':'ts_recv',
        'from_address':'from_addr', 'to_address':'to_addr',
        'transaction_count':'tx_count_blk', 'gas':'gas_limit',
        'gas_price':'gas_price', 'receipt_cumulative_gas_used':'cum_gas_used',
        'receipt_gas_used':'gas_used', 'max_fee_per_gas':'max_fee_per_gas',
        'max_priority_fee_per_gas':'priority_fee'
    })

    fmt = "%d/%m/%Y %H:%M:%S"
    df['ts_chain'] = pd.to_datetime(df['ts_chain'], format=fmt, errors='coerce')
    df['ts_recv'] = pd.to_datetime(df['ts_recv'], format=fmt, errors='coerce')

    for c in ['value_eth','tx_fee_wei','gas_price','gas_limit','gas_used','cum_gas_used','max_fee_per_gas','priority_fee']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce', downcast='float')

    records = []
    for addr, g in df.groupby('from_addr', sort=False):
        g = g.reset_index(drop=True)
        gap = g['ts_chain'].diff().dt.total_seconds().replace(0, np.nan)
        delay = (g['ts_recv'] - g['ts_chain']).dt.total_seconds()
        fee_wei = g['tx_fee_wei'] if 'tx_fee_wei' in g else g['gas_price'] * g['gas_used']
        med_gp = g['gas_price'].median()
        records.append({
            'address': addr,
            'total_tx': len(g),
            'tx_per_day': len(g) / g['ts_chain'].dt.normalize().nunique(),
            'mean_gap': gap.mean(),
            'night_ratio': (g['ts_chain'].dt.hour < 8).mean(),
            'fee_pct': (fee_wei / (g['value_eth'] * 1e18)).replace([np.inf, -np.inf], np.nan).mean(),
            'gas_premium': (g['gas_price'] / med_gp).mean(),
            'tip_ratio': (g['priority_fee'] / g['max_fee_per_gas']).replace([np.inf, -np.inf], np.nan).mean(),
            'delay_avg': delay.mean(),
            'delay_std': delay.std(ddof=0),
            'block_load': g['tx_count_blk'].iloc[0] / 500 if 'tx_count_blk' in g else 0,
            'pos_ratio': (g['cum_gas_used'] / g['cum_gas_used'].max()).mean() if 'cum_gas_used' in g else 0,
            'uniq_to': g['to_addr'].nunique(),
            'dust_ratio': (g['value_eth'] < 0.001).mean(),
            'avg_val_eth': g['value_eth'].mean()
        })

    feat = pd.DataFrame(records).fillna(0)

    for col in ['fee_pct','gas_premium','delay_avg','delay_std']:
        feat[col] = np.log1p(feat[col])
    for col in feat.select_dtypes('float').columns:
        feat[col] = feat[col].astype(np.float32)

    num_feats = feat.select_dtypes('number').columns.difference(['address'])
    X = StandardScaler().fit_transform(feat[num_feats]).astype(np.float32)

    labels = None
    try:
        import hdbscan
        labels = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10).fit_predict(X)
    except:
        pass
    if labels is None:
        best_score = -1
        for eps in np.arange(0.6, 1.6, 0.1):
            mdl = DBSCAN(eps=eps, min_samples=5).fit(X)
            labs = mdl.labels_
            k = len(set(labs)) - (1 if -1 in labs else 0)
            if k < 2: continue
            s = silhouette_score(X[labs != -1], labs[labs != -1])
            if s > best_score:
                best_score, labels = s, labs
        if labels is None:
            labels = DBSCAN(eps=1.0, min_samples=4).fit_predict(X)

    feat['cluster'] = labels
    feat['iso_flag'] = IsolationForest(contamination=0.05, random_state=0).fit_predict(X)

    q = {c: feat[c].quantile(0.9) for c in ['tx_per_day','total_tx','gas_premium','tip_ratio','dust_ratio','avg_val_eth']}
    def tag_row(r):
        if r.iso_flag == -1: return "未能识别交易模式"
        if r.gas_premium > q['gas_premium'] and r.tip_ratio > q['tip_ratio']: return "抢跑/MEV"
        if r.dust_ratio > q['dust_ratio']: return "Dusting 洗钱"
        if r.tx_per_day > q['tx_per_day']: return "高频搬砖"
        if r.avg_val_eth > q['avg_val_eth'] and r.total_tx > q['total_tx']: return "巨鲸操盘"
        return "普通用户"

    feat['behavior_tag'] = feat.apply(tag_row, axis=1)
    result = feat[['address','behavior_tag','cluster','iso_flag']]

    # 清理资源
    del df, feat, X
    gc.collect()

    # 返回 JSON 数据
    #return JSONResponse(content=result.to_dict(orient="records"))
    return JSONResponse(content={"res": result.to_dict(orient="records")})
