import numpy as np
import requests

def get_embedding(text):
    url = "http://192.168.31.48:1234/v1/embeddings"
    payload = {"model": "text-embedding-baai-bge-m3-568m", "input": text}
    res = requests.post(url, json=payload, headers={"Authorization": "Bearer sk-dummy"}).json()
    return np.array(res["data"][0]["embedding"])

def cosine_similarity(v1, v2):
    # 手动计算标准的余弦相似度
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

query = "痛风能吃豆制品吗"
doc = "《痛风与饮食管理指南》\n\n一、痛风概述\n痛风（Gout）是一种由单钠尿酸盐（MSU）晶体沉积于关节及周围组织引起的代谢性关节炎。\n其核心病理机制是血清尿酸水平长期超标（成人男性 > 420 μmol/L，女性 > 360 μmol/L），\n导致尿酸盐结晶在关节腔内沉积，诱发急性炎症反应。\n\n二、高嘌呤食物（应限制或禁止）\n痛风患者需严格控制嘌呤摄入，以下为高嘌呤食物（嘌呤含量 > 150 mg/100g）：\n- 动物内脏：肝脏、肾脏、胰腺（含嘌呤 200–400 mg/100g）\n- 海鲜类：沙丁鱼、凤尾鱼、牡蛎、贻贝、扇贝\n- 肉汤/浓汤：长时间熬制的骨汤、肉汤中嘌呤溶出浓度极高\n- 啤酒和白酒：乙醇促进尿酸合成并抑制肾脏排泄"

v_query = get_embedding(query)
v_doc = get_embedding(doc)

score = cosine_similarity(v_query, v_doc)
print(f"👉 原始 Cosine 相似度得分: {score:.4f}")