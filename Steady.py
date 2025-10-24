import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import csv
import random


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


def mainRun(knockout_data, output_file, p_lambda=0, p_alpha=1):
    data = pd.read_csv(knockout_data, '\t')
    # print(len(data))
    seed = random.randint(1, 100)
    score_1 = []
    score_2 = []
    dict_1 = []
    dict_2 = []
    # print(type(data["G1"]))
    for index in range(len(data.columns)):
        print(data.columns[index])
        data_copy = data.copy()
        y = data_copy[data_copy.columns[index]]
        y_normal = (y-np.mean(y)) / np.std(y)
        # print(type(y_normal))
        x_c = data_copy.drop(data_copy.columns[index], axis=1)
        clfx = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.0001, subsample=0.8,
                                reg_lambda=p_lambda,
                                reg_alpha=p_alpha, colsample_bylevel=0.6, colsample_bytree=0.6, seed=seed)
        clfx.fit(x_c, y_normal)
        err_1 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        # print(_importance_per)
        tmp_large = getlinks(data_copy.columns[index], x_c.columns.values, _importance_per)
        # print(tmp_large)
        score_1.append(err_1)
        dict_1.append(tmp_large)

        # local-out model
        data_copy = data.copy()
        y = data_copy[data_copy.columns[index]]
        y_normal = (y - np.mean(y)) / np.std(y)
        x_c = data_copy.drop(data_copy.columns[index], axis=1)
        clfx = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.0001, subsample=0.8,
                                reg_lambda=p_lambda,
                                reg_alpha=p_alpha, colsample_bylevel=0.6, colsample_bytree=0.6, seed=seed)
        clfx.fit(x_c, y_normal)
        err_2 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        tmp_large = getlinks(data_copy.columns[index], x_c.columns.values, _importance_per)
        score_2.append(err_2)
        dict_2.append(tmp_large)

    all_df = compute_feature_importances(score_1, score_2, dict_1, dict_2)
    all_df[['total']].to_csv(output_file, sep="\t", header=False, quoting=csv.QUOTE_NONE, escapechar=" ")

mainRun('Ecoli-80_knockouts[0].tsv', output_file='200_SS.xls')
