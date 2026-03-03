# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import lightgbm as lgb
import shap
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from plot_helper import data_anomal_filter, data_extremum, dependence_plot, force_plot, group_plot


def model_explain(data_path, label_col='label'):
    df = pd.read_csv(data_path, index_col=0)
    all_labels = df[label_col].values
    all_data = df.drop([label_col], axis=1)
    for col in all_data.select_dtypes(include=['object']).columns:
        all_data[col] = LabelEncoder().fit_transform(all_data[col])

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_labels, test_size=0.1, random_state=42)
    lgb_train = lgb.Dataset(train_data, train_label)
    lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test,
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    preds = model.predict(test_data, num_iteration=model.best_iteration)

    preds_y = [round(pred) for pred in preds]
    print('The roc of prediction is {}'.format(roc_auc_score(test_label, preds)))
    print('The acc of prediction is {}'.format(accuracy_score(test_label, preds_y)))
    print('The precision of prediction is {}'.format(precision_score(test_label, preds_y)))
    print('The recall of prediction is {}'.format(recall_score(test_label, preds_y)))
    bg_data = train_data.sample(n=min(200, train_data.shape[0]), random_state=42)
    # 传入背景数据使 TreeExplainer 使用 interventional (marginal expectation) 模式，
    # 通过边际分布替换缺失特征，打断特征间相关性，使 SHAP 值具有因果"干预"语义
    explainer = shap.TreeExplainer(model, bg_data)
    start = time.perf_counter()
    shap_values = explainer.shap_values(all_data)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.asarray(expected_value).ndim > 0:
        expected_value = np.asarray(expected_value)[1]
    end = time.perf_counter()
    print('times:', end - start)

    data_filtered, shap_values_filtered = data_anomal_filter(all_data, shap_values, 99.95)

    save_root = 'data/titanic/shap/'
    for d in ['', 'dependence', 'force', 'group']:
        os.makedirs(os.path.join(save_root, d), exist_ok=True)

    shap.summary_plot(shap_values_filtered, data_filtered, feature_names=list(all_data.columns), max_display=20, show=False)
    plt.savefig(save_root + 'summary.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('saved summary plot')

    feature_inds = np.argsort(np.abs(shap_values_filtered).mean(axis=0))
    extremum_dict = data_extremum(data_filtered, shap_values_filtered, all_data.columns)
    dependence_plot(data_filtered, shap_values_filtered, feature_inds, all_data.columns, extremum_dict, save_path=save_root + 'dependence/')
    print('saved dependence plots')

    data_filtered_df = pd.DataFrame(data_filtered, columns=all_data.columns)
    force_plot(expected_value, shap_values_filtered, data_filtered_df, save_path=save_root + 'force/', max_samples=5)
    print('saved force plots')

    group_plot(expected_value, shap_values_filtered, data_filtered_df, save_path=save_root + 'group/group.html')
    print('saved group plot')


if __name__ == '__main__':
    model_explain(data_path='data/titanic/titanic_disc.csv')


