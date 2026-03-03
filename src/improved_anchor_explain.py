# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from anchors import anchor_tabular
# from anchor import anchor_tabular
from ent_mdlp import mdlpx
import numpy as np
import pandas as pd
import pickle
import time
import lightgbm
import joblib
import glob
import os
import pathos
import multiprocessing
from multiprocessing import Manager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, classification_report


def data_preprocess(train_path):
    train_df = pd.read_csv(train_path)
    train_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    train_df = train_df.fillna(value=-1)
    print(train_df.info())
    return train_df


def data_discretize(data, data_disc_path, data_disc_dict_path):
    y = np.array(data['Survived'])
    X = data.drop(['Survived'], axis=1)
    continuous_features = ['Age', 'SibSp', 'Parch', 'Fare']
    feature_names = list(X.columns)
    continuous_features_idx = [feature_names.index(feature) for feature in continuous_features]
    for feature in continuous_features:
        if X[feature].dtypes == object:
            X[feature] = X[feature].astype(float)
    X_disc, dic, dic_all = mdlpx(X, y, continuous_features_idx, quantile=100)
    print(dic_all)
    X_disc['label'] = y
    X_disc.to_csv(data_disc_path)
    with open(data_disc_dict_path, 'wb') as f:
        pickle.dump(dic_all, f, pickle.HIGHEST_PROTOCOL)


def data_load(path):
    df = pd.read_csv(path, index_col=0)
    print(df['label'].value_counts())
    return df


def build_dataset(df, cut_path):
    y = np.array(df['label'])
    X = df.drop(['label'], axis=1)
    feature_names = list(X.columns)
    with open(cut_path, 'rb') as f:
        dic_all = pickle.load(f)
        print(dic_all)
    continuous_features = []
    continuous_names = {}
    for feature in dic_all:
        continuous_features.append(feature)
        if len(dic_all[feature][1]) == 1 and dic_all[feature][1][0] == dic_all[feature][2][0]:
            fname = '%.2f <= %s <= %.2f' % (dic_all[feature][0][0], feature, dic_all[feature][2][0])
            continuous_names[feature_names.index(feature)] = [fname]
        else:
            fname_list = ['%.2f <= %s <= %.2f' % (dic_all[feature][0][0], feature, dic_all[feature][1][0])]
            for i in range(1, len(dic_all[feature][1])):
                fname_list.append('%.2f < %s <= %.2f' % (dic_all[feature][1][i-1], feature, dic_all[feature][1][i]))
            fname_list.append('%.2f < %s <= %.2f' % (dic_all[feature][1][-1], feature, dic_all[feature][2][0]))
            continuous_names[feature_names.index(feature)] = fname_list

    # for feature in continuous_features:
    #     if X[feature].dtypes == object:
    #         X[feature] = X[feature].astype(float)

    categorical_features = [feature for feature in feature_names if feature not in continuous_features]
    for feature in categorical_features:
        # if X[feature].dtypes != object:
        X[feature] = X[feature].astype(str)
    categorical_names = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(X[feature])
        X[feature] = encoder.transform(X[feature])
        categorical_names[feature_names.index(feature)] = list(encoder.classes_)
    return X, y, categorical_features, categorical_names, continuous_features, continuous_names


def model_train(X, y, categorical_features, model_path, train=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    if train:
        model = lightgbm.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=1000, n_jobs=1, early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_logloss', categorical_feature=categorical_features)
        #list(X.columns)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
    preds = model.predict(X, num_iteration=model.best_iteration_)
    print('The accuracy is {}'.format(accuracy_score(y, preds)))
    print('The precision is {}'.format(precision_score(y, preds)))
    print('The recall is {}'.format(recall_score(y, preds)))
    return model


def model_train_overfit(X, y, categorical_features, model_path, train=True):
    if train:
        model = lightgbm.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=150, n_jobs=1, early_stopping_rounds=20)
        model.fit(X, y, eval_set=[(X, y)], eval_metric='binary_logloss', categorical_feature=categorical_features)
        # list(X.columns)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
    preds = model.predict(X, num_iteration=model.best_iteration_)
    print('The accuracy is {}'.format(accuracy_score(y, preds)))
    print('The precision is {}'.format(precision_score(y, preds)))
    print('The recall is {}'.format(recall_score(y, preds)))
    return model


def model_load(path):
    model = joblib.load(path)
    return model


def build_anchor_explainer(X, y, categorical_names, continuous_names):
    categorical_names.update(continuous_names)
    label_columns = ['Unsurvival', 'Survival']
    feature_names = X.columns
    explainer = anchor_tabular.AnchorTabularExplainer(label_columns, feature_names, X.values, categorical_names)
    return explainer, label_columns


def parallel_build_explanations_plus(target_X, explainer, model, node_nums, anchor_path):
    def batch_build_explanations(batch, state_global):
        output_file = anchor_path + 'explanation-{}'.format(multiprocessing.current_process().pid)
        print('start multiprocessing explanation:' + output_file)
        batch_explanation = {}
        with open(output_file + '.txt', 'w', encoding='utf-8') as f:
            for i in batch:
                start = time.time()
                explanation = explainer.explain_instance(target_X[i], model.predict, max_anchor_size=5, threshold=0.95, state_global=state_global, counterfactual=True)
                end = time.time()
                print('index:{}, times:{}'.format(i, end - start))
                batch_explanation[i] = explanation
                condition = ' AND '.join(explanation.names())
                print('rule:', condition, explanation.label(), explanation.precision())
                # print('counterfactual_rule:', explanation.counterfactual())
                f.write(str(i)+','+condition+','+str(explanation.label())+','+str(explanation.precision()) + '@@@' + str(explanation.counterfactual()) + '\n')
        # with open(output_file, 'wb') as f:
        #     dill.dump(batch_explanation, f)
        return batch_explanation
    target_X = target_X.values
    explanations = {}
    pool = pathos.multiprocessing.ProcessPool(nodes=node_nums)
    batch_num = node_nums
    batches = [[] for _ in range(batch_num)]
    for i in range(target_X.shape[0]):
        batches[i % batch_num].append(i)
    state_global = {}
    batch_explanations = pool.amap(batch_build_explanations, batches, [state_global for _ in range(batch_num)]).get()
    pool.close()
    pool.join()
    for batch_explanation in batch_explanations:
        explanations.update(batch_explanation)
    explanations = [explanations[key] for key in sorted(explanations.keys())]
    return explanations


def explanations_load(path):
    explanation_list = [f for f in os.listdir(path) if f.endswith('.txt')]
    indexs, rules, predictions, precisions, counterfactuals = [], [], [], [], []
    rule_set = set()
    for explanation in explanation_list:
        explanation_file = os.path.join(path, explanation)
        with open(explanation_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                # print(line)
                rules_info = line.split('@@@')[0]
                rule = rules_info.split(',')[1].split('AND')
                rule_s = tuple(set([predicate.strip() for predicate in rule]))
                if rule_s not in rule_set:
                    rule_set.add(rule_s)
                    indexs.append(rules_info.split(',')[0])
                    rules.append(rules_info.split(',')[1])
                    predictions.append(int(rules_info.split(',')[2]))
                    precisions.append(float(rules_info.split(',')[3]))
                    counterfactuals.append(eval(line.split('@@@')[1]))
    return indexs, rules, predictions, precisions, counterfactuals


def build_global_explanations(explanations_path, explainer, X, y, label_columns, result_path):
    def metric_explanation(X, prediction, rule, feature_names):
        fit_anchor = range(X.shape[0])
        use = 1
        precision_increases = []
        explanation = rule.split('AND')
        for predicate in explanation:
            predicate = predicate.strip()
            fit_anchor_tmp = fit_anchor
            if '=' in predicate and '<=' not in predicate:
                predicate_list = predicate.split('=')
                feature_name, token, feature_value = predicate_list[0].strip(), '=', predicate_list[1].strip()
                feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(feature_value)
            else:
                predicate_list = predicate.split()
                s_feature_value, s_token, feature_name, b_token, b_feature_value = float(predicate_list[0]), \
                                                                                   predicate_list[1], predicate_list[2], \
                                                                                   predicate_list[3], float(
                    predicate_list[4])
                # print(explainer.categorical_names[feature_names.index(feature_name)])
                feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(predicate)
            fit_anchor = np.intersect1d(np.where(X[:, feature_names.index(feature_name)] == feature_value), fit_anchor_tmp)
            precision_previous = np.mean(y[fit_anchor_tmp] == prediction)
            precision_now = np.mean(y[fit_anchor] == prediction)
            precision_increases.append(precision_now - precision_previous)
            if precision_now <= precision_previous:
                use = 0
        length = len(explanation)
        dataset_precision = np.mean(y[fit_anchor] == prediction)
        dataset_coverage = fit_anchor.shape[0] / float(X.shape[0])
        number = fit_anchor.shape[0]
        pred = explainer.class_names[list(np.unique(y)).index(prediction)]
        return length, dataset_precision, dataset_coverage, number, pred, use, precision_increases

    def metric_counterfactual(X, rules, feature_names):
        counterfactual_info = []
        for rule in rules:
            explanation = rule[0]
            prediction = rule[1]
            precision = rule[2]
            fit_anchor = range(X.shape[0])
            for predicate in explanation:
                fit_anchor_tmp = fit_anchor
                if '=' in predicate and '<=' not in predicate:
                    predicate_list = predicate.split('=')
                    feature_name, token, feature_value = predicate_list[0].strip(), '=', predicate_list[1].strip()
                    feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(
                        feature_value)
                else:
                    predicate_list = predicate.split()
                    s_feature_value, s_token, feature_name, b_token, b_feature_value = float(predicate_list[0]), \
                                                                                       predicate_list[1], \
                                                                                       predicate_list[2], \
                                                                                       predicate_list[3], float(
                        predicate_list[4])
                    feature_value = list(explainer.categorical_names[feature_names.index(feature_name)]).index(
                        predicate)
                fit_anchor = np.intersect1d(np.where(X[:, feature_names.index(feature_name)] == feature_value),
                                            fit_anchor_tmp)
            number = fit_anchor.shape[0]
            if number == 0:
                dataset_precision = 0.0
            else:
                dataset_precision = np.mean(y[fit_anchor] == prediction)
            cf_rule = ' AND '.join(explanation)
            counterfactual_info.append(
                '{} (precision={:.2%}, samples={}, disturb_precision={:.3f})'.format(
                    cf_rule, float(dataset_precision), number, float(precision)))
        return counterfactual_info
    print(explanations_path)
    indexs, rules, predictions, precisions, counterfactuals = explanations_load(explanations_path)

    df = pd.DataFrame()
    feature_names = list(X.columns)
    X = X.values
    lengths, dataset_precisions, dataset_coverages, conditions, preds, numbers, uses, precision_increases, counterfactual_infos = [], [], [], [], [], [], [], [], []
    for i in range(len(indexs)):
        print('index:{}'.format(i))
        length, dataset_precision, dataset_coverage, number, pred, use, precision_increase = metric_explanation(X, predictions[i], rules[i], feature_names)
        lengths.append(length)
        dataset_precisions.append(dataset_precision)
        dataset_coverages.append(dataset_coverage)
        preds.append(pred)
        numbers.append(number)
        uses.append(use)
        precision_increases.append(' → '.join(['{:+.2%}'.format(float(v)) for v in precision_increase]))
        cf_list = metric_counterfactual(X, counterfactuals[i], feature_names)
        counterfactual_infos.append(' | '.join(cf_list) if cf_list else '')

    df['rule'] = rules
    df['prediction'] = preds
    df['length'] = lengths
    df['dataset_precision'] = dataset_precisions
    df['dataset_numbers'] = numbers
    df['dataset_coverage'] = dataset_coverages
    df['precision_increase'] = precision_increases
    df['disturb_space_precision'] = precisions
    df['counterfactual_rules'] = counterfactual_infos
    df['uses'] = uses
    df = df.drop(['uses'], axis=1)
    df = df.drop_duplicates(['rule'])
    df.sort_values(by=['dataset_precision', 'dataset_coverage', 'length'], ascending=(False, False, True), inplace=True)
    for label in label_columns:
        df[df['prediction'] == label].to_csv(result_path+'{}.csv'.format(label), encoding='utf-8-sig')


# X：原始样本的index列表
def build_learner(explanations, model, target_X, class_names, X):
    def metric_explanation(X, x, explanation, model):
        fit_anchor = np.where(
            np.all(target_X[X][:, explanation.features()] == target_X[x][explanation.features()], axis=1))[0]
        length = len(explanation.names())
        precision = np.mean(
            model.predict(target_X[X][fit_anchor]) == model.predict(target_X[x].reshape(1, -1)))
        coverage = fit_anchor.shape[0] / float(len(X))
        condition = ' AND '.join(explanation.names())
        pred = class_names[model.predict(target_X[x].reshape(1, -1))[0]]
        return length, precision, coverage, condition, pred

    # X：原始样本的index列表
    def metric(explanations, model, X):
        df = pd.DataFrame()
        lengths = []
        precisions = []
        coverages = []
        conditions = []
        preds = []
        for x in X:
            length, precision, coverage, condition, pred = metric_explanation(X=X,
                                                                              x=x,
                                                                              explanation=explanations[x],
                                                                              model=model)
            lengths.append(length)
            precisions.append(precision)
            coverages.append(coverage)
            conditions.append(condition)
            preds.append(pred)
        df['rule'] = X
        df['length'] = lengths
        df['precision'] = precisions
        df['coverage'] = coverages
        df['condition'] = conditions
        df['pred'] = preds
        return df

    learner = []
    default_pred = 0
    while True:
        # 在当前样本x中计算规则集合rules中每个规则的覆盖率和准确率
        # 选择最优的rule(i)加入候选模型，rules移除rule
        # X中移除正确分类的样本
        # 样本全部分类完成 或者 当前top 1规则准确率小于默认类别准确率
        if len(X) == 0:
            learner.append(-1)
            break
        X_pred = model.predict(target_X[X])
        default_pred = np.argmax(np.bincount(X_pred))
        default_precision = np.mean(X_pred == default_pred)
        df = metric(explanations, model, X)
        # 初筛条件 覆盖率大于等于0.01
        X = np.setdiff1d(X, df[df.coverage < 0.01].rule.values)
        df = df[df.coverage >= 0.01]
        if len(df) == 0:
            learner.append(-1)
            break
        df.sort_values(by=['precision', 'coverage', 'length'], ascending=(False, False, True), inplace=True)
        if df.head(1).precision.values[0] <= default_precision:
            learner.append(-1)
            break
        best_explanation = df.head(1).rule.values[0]
        learner.append(best_explanation)
        fit_anchor = np.where(
            np.all(
                target_X[:, explanations[best_explanation].features()] ==
                target_X[best_explanation][explanations[best_explanation].features()],
                axis=1))[0]
        print('best rule index:', best_explanation)
        print('matched samples:', len(fit_anchor))
        X = np.setdiff1d(X, fit_anchor)
        print('remaining samples:', len(X))
        del df
    return learner, default_pred


def metric_learner(learner, X, target_X, explanations, model, default_pred):
    explainer_pred = np.zeros(len(X))
    model_pred = model.predict(target_X[X])
    X_remaining = X.copy()
    for rule in learner:
        if rule == -1:
            explainer_pred[X_remaining] = default_pred
            break
        else:
            fit_anchor = np.where(np.all(
                target_X[:, explanations[rule].features()] == target_X[rule][
                    explanations[rule].features()],
                axis=1))[0]
            matched = np.intersect1d(fit_anchor, X_remaining)
            explainer_pred[matched] = model.predict(target_X[rule].reshape(1, -1))
            X_remaining = np.setdiff1d(X_remaining, fit_anchor)
    precision = np.mean(explainer_pred == model_pred)
    print('learner precision:', precision)
    return precision


def parse_learner(learner, explanations, default_pred, class_names, model, target_X):
    print('\n===== Global Rule Model =====')
    for i, rule in enumerate(learner):
        if rule == -1:
            print('Rule {}: Else => {}'.format(i + 1, class_names[default_pred]))
        else:
            pred = class_names[model.predict(target_X[rule].reshape(1, -1))[0]]
            print('Rule {}: {} => {}'.format(
                i + 1,
                ' AND '.join(explanations[rule].names()),
                pred))
    print('=============================\n')


if __name__ == '__main__':
    train_path = 'data/titanic/train.csv'
    test_path = 'data/titanic/test.csv'

    data1_disc_path = 'data/titanic/titanic_disc.csv'
    data1_disc_dict_path = 'data/titanic/titanic_disc_dict.pkl'

    if not os.path.exists(data1_disc_path) or not os.path.exists(data1_disc_dict_path):
        print('离散化文件不存在，开始预处理...')
        df = data_preprocess(train_path)
        data_discretize(df, data1_disc_path, data1_disc_dict_path)

    model_path = 'model/lgb_titanic.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = data_load(data1_disc_path)
    X, y, categorical_features, categorical_names, continuous_features, continuous_names = build_dataset(df, data1_disc_dict_path)
    model = model_train(X, y, categorical_features, model_path)

    anchor_path = 'data/titanic/anchor/rule/'
    model = model_load(model_path)
    explainer, label_columns = build_anchor_explainer(X, y, categorical_names, continuous_names)

    os.makedirs(anchor_path, exist_ok=True)
    for f in glob.glob(os.path.join(anchor_path, '*')):
        os.remove(f)
    start = time.time()
    node_nums = min(8, os.cpu_count() or 1)
    explanations = parallel_build_explanations_plus(X, explainer, model, node_nums, anchor_path)
    end = time.time()
    print('all time consume:{}'.format(end-start))
    result_path = 'data/titanic/anchor/result/'
    os.makedirs(result_path, exist_ok=True)
    for f in glob.glob(os.path.join(result_path, '*')):
        os.remove(f)
    build_global_explanations(anchor_path, explainer, X, y, label_columns, result_path)

    # 贪心规则选择（对应论文 §4.5.3 SP 全局解释）
    target_X = X.values
    sample_indices = np.arange(target_X.shape[0])
    learner, default_pred = build_learner(explanations, model, target_X, label_columns, sample_indices)
    # 评估规则模型
    metric_learner(learner, np.arange(target_X.shape[0]), target_X, explanations, model, default_pred)
    # 打印可读规则
    parse_learner(learner, explanations, default_pred, label_columns, model, target_X)

