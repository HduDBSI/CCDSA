"""
Updated on Dec 20, 2020

create ml-1m dataset

@author: Ziyao Geng(zggzy1996@163.com)
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40, test_neg_num=100,
                         time_slot=3, special_time_split=True, min_count=5):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """

    # create datasetz
    def df_to_list(data):
        # 将序列填充为长为40(maxlen)的序列
        return [data['user_id'].values,
                pad_sequences(data['hist'], maxlen=maxlen),
                pad_sequences(data['time_list'], maxlen=maxlen),
                data['time'].values,
                pad_sequences(data['w_list'], maxlen=maxlen),
                data['w'].values,
                pad_sequences(data['h_list'], maxlen=maxlen),
                data['h'].values,
                pad_sequences(data['c_list'], maxlen=maxlen),
                data['c'].values, data['c_neg'].values,
                data['pos_item'].values, data['neg_item'].values]

    print('==========Data Preprocess Start=============')
    random.seed(10)
    names = ['user_id', 'item_id', 'label', 'Timestamp', 'lat', 'lon','weekday', 'hour', 'month', 'category']
    data_df = pd.read_csv(file, header=None, sep='\t', names=names)

    # 过滤出现次数小于5次的物品
    p = data_df.groupby('item_id')['user_id'].count().reset_index().rename(columns={'user_id': 'item_count'})
    data_df = pd.merge(data_df, p, how='left', on='item_id')
    data_df = data_df[data_df['item_count'] >= min_count].drop(['item_count'], axis=1)
    # 过滤出现次数小于5次的用户
    p = data_df.groupby('user_id')['item_id'].count().reset_index().rename(columns={'item_id': 'user_count'})
    data_df = pd.merge(data_df, p, how='left', on='user_id')
    data_df = data_df[data_df['user_count'] >= min_count].drop(['user_count'], axis=1)

    # ReMap item ids
    item_unique = data_df['item_id'].unique().tolist()
    item_map = dict(zip(item_unique, range(1, len(item_unique) + 1)))
    data_df['item_id'] = data_df['item_id'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data_df['user_id'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    data_df['user_id'] = data_df['user_id'].apply(lambda x: user_map[x])

    data_df = data_df.sort_values(by=['user_id', 'Timestamp']).reset_index(drop=True)

    # creat lat_lon_map and category_map
    lat_lon_df = pd.DataFrame({'lat':data_df.groupby('item_id')['lat'].mean(),
                               'lon':data_df.groupby('item_id')['lon'].mean()})
    lat_lon_df.loc[0] = [0,0]
    lat_lon_df.sort_index(inplace=True)
    lat_lon_map = lat_lon_df.values

    category_map = data_df[['item_id','category']].sort_values('item_id').drop_duplicates('item_id', keep='first').set_index('item_id').to_dict()

    # slot time
    def get_time_slot(t):
        if special_time_split:
            if 6 <= t <= 12:
                return 0
            elif 12 < t <= 18:
                return 1
            else:
                return 2
        else:
            return int(t*time_slot/24)

    data_df['hour'] = data_df['hour'].apply(lambda x:get_time_slot(x))

    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, test_data, new_test_data = [], [], []
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id', 'Timestamp','weekday',
                                     'hour', 'month', 'category']].groupby('user_id')):
        pos_list = df['item_id'].tolist()
        time_list = df['Timestamp'].tolist()
        w_list = df['weekday'].tolist()
        h_list = df['hour'].tolist()
        c_list = df['category'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
                return neg

        def get_category(item_id):
            return category_map['category'][item_id]

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            time_i = time_list[:i]
            w_i = w_list[:i]
            h_i = h_list[:i]
            c_i = c_list[:i]
            if i == len(pos_list) - 1:
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i,
                                      time_i, time_list[i],
                                      w_i, w_list[i],
                                      h_i, h_list[i],
                                      c_i, c_list[i], get_category(neg),
                                      pos_list[i], neg])
                if pos_list[i] not in hist_i:
                    for neg in neg_list[i:]:
                        new_test_data.append([user_id, hist_i,
                                              time_i, time_list[i],
                                              w_i, w_list[i],
                                              h_i, h_list[i],
                                              c_i, c_list[i], get_category(neg),
                                              pos_list[i], neg])
            else:
                train_data.append([user_id, hist_i,
                                   time_i, time_list[i],
                                   w_i, w_list[i],
                                   h_i, h_list[i],
                                   c_i, c_list[i], get_category(neg_list[i]),
                                   pos_list[i], neg_list[i]])

    # feature columns
    user_num, item_num, c_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1, data_df['category'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim),
                       sparseFeature('category', c_num, embed_dim)]
    # shuffle 将序列的所有元素随机排序。
    random.shuffle(train_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist',
                                              'time_list', 'time',
                                              'w_list', 'w',
                                              'h_list', 'h',
                                              'c_list', 'c', 'c_neg',
                                              'pos_item', 'neg_item'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist',
                                            'time_list', 'time',
                                            'w_list', 'w',
                                            'h_list', 'h',
                                            'c_list', 'c', 'c_neg',
                                            'pos_item', 'neg_item'])
    new_test = pd.DataFrame(new_test_data, columns=['user_id', 'hist',
                                                    'time_list', 'time',
                                                    'w_list', 'w',
                                                    'h_list', 'h',
                                                    'c_list', 'c', 'c_neg',
                                                    'pos_item', 'neg_item'])
    print('==================Padding===================')
    train_X = df_to_list(train)
    test_X = df_to_list(test)
    new_test_X = df_to_list(new_test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, test_X, new_test_X, lat_lon_map


# create_ml_1m_dataset('../dataset/ml-1m/ratings.dat')