"""
Updated on Dec 20, 2020

train SASRec model

@author: Ziyao Geng(zggzy1996@163.com)
"""
import argparse
import csv
import os
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from CSASRec_model import SASRec
from CSASRec_evaluate import *
from CSASRec_utils import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--len_Seq', type=int, default=200)
    parser.add_argument('--neg_sample', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--l2_lambda', type=float, default=1e-6)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--ffn_hidden_unit', type=int, default=64)
    parser.add_argument('--time_slot', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--norm_training', type=bool, default=True)
    parser.add_argument('--causality', type=bool, default=True)
    parser.add_argument('--special_time_split', type=bool, default=True)
    parser.add_argument('--data_path', type=str,
                        # default='/home/wf/shenyi/Data/weeplaces/output/SA_weeplace/SA_DT_C.data')
                        # default='/home/shenyi/Data/dataset_tsmc2014/TKY/TKY_DT_C.data')
                        default='/home/wf/shenyi/Data/dataset_tsmc2014/NYC/NYC_DT_C.data') # len_Seq=210 block 2
                        # default='/home/wf/shenyi/Data/dataset_tsmc2014/TKY/TKY_DT_C.data') # len_Seq=250 block 2
                        # default='/home/wf/shenyi/Data/weeplaces/output/NY_weeplace/NYC_DT_C.data')  # len_Seq=121 block 2
                        # default='/home/wf/shenyi/Data/weeplaces/output/SA_weeplace/SA_DT_C.data') # len_Seq=54 block 1
    parser.add_argument('--data_name', type=str, default='NYC') # WSF/WNY/TKY/NY
    parser.add_argument('--min_count', type=int, default=5)
    return parser.parse_args()

# nohup python -u CSASRec_train.py --len_Seq=54 --block=1 --min_count=15 --neg_sample=90 > result_NYC_1.txt 2>&1 &

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # ========================= Hyper Parameters =======================
    args = parse_args()
    file = args.data_path
    file_name = args.data_name
    maxlen = args.len_Seq
    test_neg_num = args.neg_sample
    embed_dim = args.emb_size
    blocks = args.blocks
    num_heads = args.num_heads
    ffn_hidden_unit = args.ffn_hidden_unit
    dropout = args.dropout
    norm_training = args.norm_training
    causality = args.causality
    embed_reg = args.l2_lambda  # 1e-6
    K = [1, 3, 5, 10]
    learning_rate = args.learning_rate
    epochs = args.num_epochs
    batch_size = args.batch_size
    time_slot = args.time_slot
    special_time_split = args.special_time_split
    min_count = args.min_count

    # ========================== Create dataset =======================
    feature_columns, train, test, new_test, lat_lon_map = create_ml_1m_dataset(file, 1, embed_dim, maxlen,
                                                                               test_neg_num, time_slot,
                                                                               special_time_split, min_count)

    # ============================Build Model==========================
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model = SASRec(feature_columns, lat_lon_map, blocks, num_heads, ffn_hidden_unit, dropout,
    #                    maxlen, norm_training, causality, embed_reg)
    #     model.summary()
    #     # =========================Compile============================
    #     model.compile(optimizer=Adam(learning_rate=learning_rate))

    model = SASRec(feature_columns, lat_lon_map, time_slot, blocks, num_heads, ffn_hidden_unit, dropout,
                   maxlen, norm_training, causality, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate), run_eagerly=True)

    results = []
    new_results = []
    # 第一列代表epoch
    results.append([])
    new_results.append([])
    results_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_results_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(K)):
        # 每一个K有两列,分别为HR和NDCG
        results.append([])
        results.append([])
        new_results.append([])
        new_results.append([])
    count = 0
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train,
            epochs=1,
            batch_size=batch_size,
        )
        t2 = time()
        if epoch % 5 == 0:
            results[0].append(epoch)
            new_results[0].append(epoch)
            print('=================Next recommendation===================')
            for i in range(len(K)):
                hit_rate, ndcg, _ = evaluate_model(model, test, K[i])
                print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, TOP@%d'
                      % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, K[i]))
                results[2 * i + 1].append(hit_rate)
                results[(2 * i + 2)].append(ndcg)
            # 如果最后一行的hr@10最大则更新
            if results[7][-1] > results_max[7]:
                for i in range(len(results)):
                    results_max[i] = results[i][-1]
                count = 0
            else:
                count = count + 1
            print('=================Next New recommendation===================')
            for i in range(len(K)):
                new_hit_rate, new_ndcg, _ = evaluate_model(model, new_test, K[i])
                print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, TOP@%d'
                      % (epoch, t2 - t1, time() - t2, new_hit_rate, new_ndcg, K[i]))
                new_results[2 * i + 1].append(new_hit_rate)
                new_results[(2 * i + 2)].append(new_ndcg)
                # 如果最后一行的hr@10最大则更新
            if new_results[7][-1] > new_results_max[7]:
                for i in range(len(new_results)):
                    new_results_max[i] = new_results[i][-1]
            if count >= 10:
                break


            def write_log(r, rm, is_new=''):
                # write log
                df = pd.DataFrame(r).T
                df.columns = ['Iteration', 'HR@1', 'NDCG@1',
                              'HR@3', 'NDCG@3',
                              'HR@5', 'NDCG@5',
                              'HR@10', 'NDCG@10']
                df.to_csv('log/CSASRec_{}_log_d_{}_maxlen_{}_dim_{}.csv'.format(
                    is_new, file_name, maxlen, embed_dim), index=False)
                with open('log/CSASRec_{}_maxLog_d_{}_maxlen_{}_dim_{}.csv'.format(
                        is_new, file_name, maxlen, embed_dim), 'w', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(rm)


            write_log(results, results_max)
            # # 表示记录next new
            # write_log(new_results,new_results_max, is_new='new')
