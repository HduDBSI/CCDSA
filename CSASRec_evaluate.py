"""
Updated on Dec 20, 2020

evaluate model

@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np
import pandas as pd


# def evaluate_model(model, test, K):
#     """
#     evaluate model
#     :param model: model
#     :param test: test set
#     :param K: top K
#     :return: hit rate, ndcg
#     """
#     pred_y = - model.predict(test)
#     rank = pred_y.argsort().argsort()[:, 0]
#     hr, ndcg = 0.0, 0.0
#     for r in rank:
#         if r < K:
#             hr += 1
#             ndcg += 1 / np.log2(r + 2)
#     return hr / len(rank), ndcg / len(rank)


def getHit(df):
    """
    calculate hit rate
    :return:
    """
    if sum(df['pred']) < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    """
    calculate NDCG
    :return:
    """
    if sum(df['pred']) < _K:
        return 1 / np.log2(sum(df['pred']) + 2)
    else:
        return 0.


def getMRR(df):
    """
    calculate MRR
    :return:
    """
    return 1 / (sum(df['pred']) + 1)


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    global _K
    _K = K
    test_X = test
    # predict
    pos_score, neg_score = model.predict(test_X)
    # create dataframe
    test_df = pd.DataFrame(test_X[0], columns=['user_id'])
    # if mode == 'inner', pos score < neg score, pred = 1
    test_df['pred'] = (pos_score <= neg_score).astype(np.int32)
    # groupby
    tg = test_df.groupby('user_id')
    # calculate hit
    hit_rate = tg.apply(getHit).mean()
    # calculate ndcg
    ndcg = tg.apply(getNDCG).mean()
    # calculate mrr
    mrr = tg.apply(getMRR).mean()
    return hit_rate, ndcg, mrr