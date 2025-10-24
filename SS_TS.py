import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import csv
import random
import time
import warnings
warnings.filterwarnings("ignore")

def single_col_timeseries(scol, split, timelag, maxlag):
    slen = int(len(scol)/split)
    res_col = pd.Series()
    for index in range(0,split):
        tmp = scol[int(index*slen+timelag):(slen+index*slen+timelag-maxlag)]
        res_col = res_col.append(tmp, ignore_index=True)
    # print(res_col)
    return res_col


def invert_expression_timeseries(exp_mat, split, maxlag, pan=0):
    df = pd.DataFrame()
    all_mean = np.mean(exp_mat.values)
    all_std = np.std(exp_mat.values)
    for index in range(0, len(exp_mat.columns)):
        sname=exp_mat.columns[index];
        #df[sname]=exp_mat[sname]
        for jindex in range(0+pan,maxlag+pan):
            #use random to change the sequence
            df[sname] = single_col_timeseries(exp_mat[sname],split,jindex,maxlag)
    # print(df)
    return df


def getlinks(target_name, name, importance_, inverse=False):
    feature_imp=pd.DataFrame(importance_, index=name, columns=['imp'])
    feature_large_set = {}
    for i in range(0, len(feature_imp.index)):
        tmp_name=feature_imp.index[i].split('_')
        if tmp_name[0] != target_name:
            if not inverse:
                if (tmp_name[0]+"\t"+target_name) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = max(feature_large_set[tmp_name[0] + "\t" + target_name],tf_score)
            else:
                if (target_name + "\t" + tmp_name[0]) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[target_name + "\t" + tmp_name[0]] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']

                    feature_large_set[target_name + "\t" + tmp_name[0]] = max(
                        feature_large_set[target_name + "\t" + tmp_name[0]], tf_score)
    return feature_large_set


def compute_feature_importances(score_1, score_2, dicts_1, dicts_2):
    dict_all_1 = {}
    dict_all_2 = {}
    score_1 = 1-score_1 / sum(score_1)
    score_2 = 1-score_2 / sum(score_2)
    for i in range(len(score_1)):
        tmp_dict = dicts_1[i]
        for key in tmp_dict:
            tmp_dict[key] = tmp_dict[key]*score_1[i]
        dict_all_1.update(tmp_dict)

    for i in range(len(score_2)):
        tmp_dict = dicts_2[i]
        for key in tmp_dict:
            tmp_dict[key] = tmp_dict[key]*score_2[i]
        dict_all_2.update(tmp_dict)

    d1 = pd.DataFrame.from_dict(dict_all_1, orient='index')
    d1.columns = ["score_1"]
    d2 = pd.DataFrame.from_dict(dict_all_2, orient='index')
    d2.columns = ["score_2"]

    all_df = d1.join(d2)
    all_df['total'] = np.sqrt(all_df["score_1"] * all_df["score_2"])

    return all_df


def KN_TS_mainRun(knockout, timeseries, samples, outputfile='my_output.xls', p_lambda=0, p_alpha=1, timeLag=1):
    # 两组数据共用变量
    seed = random.randint(1, 100)
    score_1 = []
    score_2 = []
    dict_1 = []
    dict_2 = []
    # 两组数据
    KN_data = pd.read_csv(knockout, delimiter='\t')
    TS_data = pd.read_csv(timeseries, delimiter='\t')
    # print(TS_data)
    for index in range(len(TS_data.columns)):
        print(KN_data.columns[index])
        # local - in
        KN_data_copy = KN_data.copy()
        TS_data_copy = TS_data.copy()

        KN_y = KN_data_copy[KN_data_copy.columns[index]]
        KN_y_normal = (KN_y-np.mean(KN_y)) / np.std(KN_y)
        KN_x_c = KN_data_copy.drop(KN_data_copy.columns[index], axis=1)

        TS_y = single_col_timeseries(TS_data[TS_data.columns[index]], samples, timeLag, timeLag)
        TS_y_normal = (TS_y-np.mean(TS_y)) / np.std(TS_y)
        TS_x_c = invert_expression_timeseries(TS_data_copy, samples, timeLag)
        TS_x_c = TS_x_c.drop(TS_x_c.columns[index], axis=1)

        y_normal = KN_y_normal.append(TS_y_normal)
        x_c = KN_x_c.append(TS_x_c)

        # y_normal = KN_y_normal
        # x_c = KN_x_c

        clfx = xgb.XGBRegressor(max_depth=3, n_estimators=2000, learning_rate=0.0001, subsample=0.8,
                                reg_lambda=p_lambda,
                                reg_alpha=p_alpha, colsample_bylevel=0.6, colsample_bytree=0.6, seed=seed)
        clfx.fit(x_c, y_normal)
        err_1 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        # print(_importance_per)
        tmp_large = getlinks(TS_data_copy.columns[index], x_c.columns.values, _importance_per)
        # print(tmp_large)
        score_1.append(err_1)
        dict_1.append(tmp_large)

        # local - out
        KN_data_copy = KN_data.copy()
        TS_data_copy = TS_data.copy()

        KN_y = KN_data_copy[KN_data_copy.columns[index]]
        KN_y_normal = (KN_y - np.mean(KN_y)) / np.std(KN_y)
        KN_x_c = KN_data_copy.drop(KN_data_copy.columns[index], axis=1)

        TS_y = single_col_timeseries(TS_data_copy[TS_data_copy.columns[index]], samples, 0, timeLag)
        TS_y_normal = (TS_y - np.mean(TS_y)) / np.std(TS_y)
        TS_x_c = invert_expression_timeseries(TS_data_copy, samples, timeLag, 1)
        TS_x_c = TS_x_c.drop(TS_x_c.columns[index], axis=1)

        y_normal = KN_y_normal.append(TS_y_normal)
        x_c = KN_x_c.append(TS_x_c)

        clfx = xgb.XGBRegressor(max_depth=3, n_estimators=2000, learning_rate=0.0001, subsample=0.8,
                                reg_lambda=p_lambda,
                                reg_alpha=p_alpha, colsample_bylevel=0.6, colsample_bytree=0.6, seed=seed)
        clfx.fit(x_c, y_normal)
        err_2 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        tmp_large = getlinks(TS_data_copy.columns[index], x_c.columns.values, _importance_per)
        score_2.append(err_2)
        dict_2.append(tmp_large)

    all_df = compute_feature_importances(score_1, score_2, dict_1, dict_2)
    all_df[['total']].to_csv(outputfile, sep="\t", header=False, quoting=csv.QUOTE_NONE, escapechar=" ")


if __name__ == '__main__':
    start_time = time.time()
    KN_TS_mainRun("Ecoli-80_knockouts[0].tsv", "Ecoli-80_timeseries[0].tsv", 10, outputfile="80_2.xls")
    end_time = time.time()
    print(end_time - start_time)
