# -*- coding: utf-8 -*-
"""LightGBM 二分类 + TreeSHAP 解释流水线。

输入: train/val/test 三段 DataFrame
输出: save_root 下生成
    - feature_importance.csv     特征重要性表
    - summary.png                SHAP summary plot
    - dependence/*.png           前 N 个特征的 dependence plot
    - force/*.png                若干样本的 force plot
    - group/group.html           交互式 group plot
"""


from pathlib import Path
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
from joblib import Parallel, delayed
from plot_helper import data_anomal_filter, data_extremum, dependence_plot, force_plot, group_plot


def _save_feature_importance_report(feature_data, shap_values, save_root):
    """导出 SHAP 特征重要性表。

    输出列含义:
        rank             按 mean_abs_shap 降序的排名
        feature          特征名 (类别列已 target encoding 成单列数值)
        mean_abs_shap    |SHAP| 的均值，衡量该特征对预测的整体影响幅度
        mean_shap        SHAP 的均值，衡量该特征整体上把预测推高(>0)还是拉低(<0)
        std_shap         SHAP 的标准差，衡量该特征影响在样本间的分歧程度
        pos_ratio        正向 SHAP 样本占比
        neg_ratio        负向 SHAP 样本占比
        importance_pct   mean_abs_shap 占总重要性的比例
        cumulative_pct   按排名累计占比，用来挑 top-K 特征做后续分析
    """
    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": list(feature_data.columns),
        "mean_abs_shap": mean_abs,
        "mean_shap": shap_values.mean(axis=0),
        "std_shap": shap_values.std(axis=0),
        "pos_ratio": (shap_values > 0).mean(axis=0),
        "neg_ratio": (shap_values < 0).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    importance_df["importance_pct"] = importance_df["mean_abs_shap"] / mean_abs.sum()
    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum()

    save_dir = Path(save_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(save_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")


def _report_binary_metrics(split_name, y_true, y_score, threshold=0.5):
    """统一打印二分类评估指标。"""
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }
    metrics_str = " | ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    print(f"[{split_name}] {metrics_str}")


def _fit_target_encoder(train_df, col, label_col, smoothing):
    """带平滑的 target encoding，返回 {category: encoded_value} 与全局均值。

    encoded(c) = (sum_y_c + smoothing * global_mean) / (count_c + smoothing)
    smoothing 越大，小样本类别越向全局均值收缩，防过拟合；常用 5~20。
    """
    global_mean = float(train_df[label_col].mean())
    # dropna=False 让 NaN 自成一类，与"未见类别"区分开
    grouped = train_df.groupby(col, dropna=False)[label_col].agg(['sum', 'count'])
    encoded = (grouped['sum'] + smoothing * global_mean) / (grouped['count'] + smoothing)
    return encoded.to_dict(), global_mean


def model_explain(train_df, val_df, test_df, feature_cols, label_col, save_root,
                  categorical_cols=(), te_smoothing=10,
                  shap_filter=None, shap_sample_n=20000, top_n_dependence=20):
    """训练 LightGBM 并生成 TreeSHAP 解释图。

    防数据泄漏要点:
        - target encoding 的映射只在 train_df 上 fit，val/test 出现新类别 / NaN 用全局均值兜底
        - early_stopping 用 val_df，不能用 test_df

    Args:
        train_df: 训练集 DataFrame。
        val_df: 验证集 DataFrame，用于 early stopping。
        test_df: 测试集 DataFrame，用于最终评估。
        feature_cols: 特征列名列表。
        label_col: 标签列名 (二分类 0/1)。
        save_root: 输出目录，函数会写入 CSV / 图片。
        categorical_cols: 类别特征列名，会被 smoothing target encoding 编码成单列数值。
            优于 one-hot: 无共线性、每个原始特征只对应一列 SHAP、解释更直观。
            优于整数顺序编码: 编码值本身就是"该类别下 y 的平滑均值"，自动按真实信号排序，
            不会强加错误的大小关系 (例如 Embarked=C/Q/S 不会被人为编成 0/1/2)。
            其余列按连续值处理 (NaN → -1)。
        te_smoothing: target encoding 平滑系数，越大越向全局均值收缩。低基数 (<10 类) 用 5~10，
            高基数 (城市/SKU) 用 20~50。
        shap_filter: 可选，callable(DataFrame) -> bool Series，
            筛选参与 SHAP 计算的样本子集 (例如只解释某类用户)。None 表示不过滤。
        shap_sample_n: SHAP 采样上限，控制计算量；样本数小于该值时全用。
        top_n_dependence: dependence plot 只画前 N 个最重要特征，避免几十张图。
    """
    save_dir = Path(save_root)
    categorical_cols = list(categorical_cols)

    # target encoding 只在 train 上 fit，避免 val/test 标签信息泄漏到编码里
    te_maps = {}
    te_defaults = {}
    for col in categorical_cols:
        mapping, global_mean = _fit_target_encoder(train_df, col, label_col, te_smoothing)
        te_maps[col] = mapping
        te_defaults[col] = global_mean
        print('target encoding [{}]: {} -> mean {:.4f}'.format(col, mapping, global_mean))

    def _prepare(df):
        """类别列做 target encoding；连续列 NaN 填 -1。返回 (X, y)。"""
        X = df[feature_cols].copy()
        for col in feature_cols:
            if col in te_maps:
                # map 时未见类别会变 NaN，再 fillna 到全局均值
                # 注意 NaN 在 te_maps 里也是一个 key (来自 dropna=False groupby)，
                # 但 Series.map 不会把 NaN 当成可查 key，所以 NaN 一律走 fillna 兜底
                X[col] = X[col].map(te_maps[col]).fillna(te_defaults[col]).astype(np.float64)
            else:
                # 连续数值列：-1 作为 NaN 哨兵 (前提是业务特征非负)，
                # 让 NaN 在树里有稳定的、可被 interventional SHAP 归因的路径
                X[col] = X[col].fillna(-1)
        return X, df[label_col].values

    X_train, train_label = _prepare(train_df)
    X_val, val_label = _prepare(val_df)
    X_test, test_label = _prepare(test_df)

    # 拼接全量供后续 SHAP 采样池使用：解释的是模型在所有数据上的归因，不只 test
    all_data = pd.concat([X_train, X_val, X_test], ignore_index=True)

    lgb_train = lgb.Dataset(X_train, train_label)
    lgb_val = lgb.Dataset(X_val, val_label, reference=lgb_train)
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
        # -1 同时静默 info 与 warning；0 只静默 info，warning 仍会刷屏
        # 小数据集 + num_leaves 偏大时会反复出 "No further splits with positive gain"，
        # 这只是树提前停止生长，不影响正确性，对工程使用无意义
        'verbosity': -1,
    }
    # valid_sets 必须是 val 集，原版用 test 集做 early stopping 等同于数据泄漏
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_val,
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    _report_binary_metrics("val", val_label, val_preds)
    _report_binary_metrics("test", test_label, test_preds)

    # 构造 SHAP 采样池：可选条件过滤 + 上限采样，控制 TreeSHAP 计算耗时
    pool = all_data
    if shap_filter is not None:
        mask = np.asarray(shap_filter(pool), dtype=bool)
        pool = pool[mask].reset_index(drop=True)
        print('SHAP filter: {} / {} samples matched'.format(len(pool), len(all_data)))

    shap_data = pool.sample(n=min(shap_sample_n, len(pool)), random_state=42).reset_index(drop=True)
    print('SHAP sample size: {} / {}'.format(len(shap_data), len(pool)))

    bg_data = X_train.sample(n=min(200, X_train.shape[0]), random_state=42)
    # 传入背景数据使 TreeExplainer 使用 interventional (marginal expectation) 模式，
    # 通过边际分布替换缺失特征，打断特征间相关性，使 SHAP 值具有因果"干预"语义。
    # 这条路径要求所有特征都是数值连续切分 → 上面已把类别列 target encoding 成单列浮点。
    explainer = shap.TreeExplainer(model, bg_data)
    start = time.perf_counter()

    # 按 CPU 核数把样本切块并行；TreeSHAP 单样本耗时随树数线性增长，并行收益明显
    n_workers = max(os.cpu_count() - 1, 1)
    chunks = np.array_split(shap_data, n_workers)
    results = Parallel(n_jobs=n_workers)(
        delayed(explainer.shap_values)(chunk) for chunk in chunks
    )
    # 二分类时 shap_values 可能是 list[ndarray]（旧版返回正负类两份），统一取正类
    shap_values = np.concatenate(
        [r[1] if isinstance(r, list) else r for r in results], axis=0
    )

    # expected_value 同上，旧版可能返回两类的基线，取正类基线
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.asarray(expected_value).ndim > 0:
        expected_value = np.asarray(expected_value)[1]
    end = time.perf_counter()
    print('times:', end - start)

    _save_feature_importance_report(shap_data, shap_values, save_dir)
    # argsort 升序：尾部 N 个 = 重要性最高的 N 个
    # 注意：排名以未过滤的 shap_values 为准，与 feature_importance.csv 保持一致
    feature_inds = np.argsort(np.abs(shap_values).mean(axis=0))
    # 按各列 99.95 分位剔除极端样本，避免 summary/dependence 图坐标轴被尾部样本拉伸
    data_filtered, shap_values_filtered = data_anomal_filter(shap_data, shap_values, 99.95)

    for sub in ['', 'dependence', 'force', 'group']:
        (save_dir / sub).mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values_filtered, data_filtered, feature_names=list(shap_data.columns), max_display=20, show=False)
    plt.savefig(save_dir / 'summary.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('saved summary plot')

    extremum_dict = data_extremum(data_filtered, shap_values_filtered, shap_data.columns)
    top_n = min(top_n_dependence, len(feature_inds))
    # plot_helper 用字符串拼接 (save_path + '_' + feature)，必须以分隔符结尾
    dependence_dir = os.path.join(str(save_dir / 'dependence'), '')
    dependence_plot(data_filtered, shap_values_filtered, feature_inds[-top_n:], shap_data.columns, extremum_dict, save_path=dependence_dir)
    print('saved dependence plots')

    # data_anomal_filter 返回 ndarray，下游 force_plot/group_plot 需要 DataFrame
    data_filtered_df = pd.DataFrame(data_filtered, columns=shap_data.columns)
    force_dir = os.path.join(str(save_dir / 'force'), '')
    force_plot(expected_value, shap_values_filtered, data_filtered_df, save_path=force_dir, max_samples=5)
    print('saved force plots')

    group_plot(expected_value, shap_values_filtered, data_filtered_df, save_path=str(save_dir / 'group' / 'group.html'))
    print('saved group plot')


if __name__ == '__main__':
    df = pd.read_csv('data/titanic/train.csv')
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    label_col = 'Survived'
    feature_cols = [c for c in df.columns if c != label_col]

    # 先 80/20 切出 test，再从 train 切出 10% 作 val (整体大致 72/8/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Pclass (1/2/3 等舱) 是 ordinal，越小越高贵，直接当数值用，不进 categorical_cols
    # Sex / Embarked 是 nominal (无序)，走 target encoding 转单列数值
    # shap_filter 用法示例: lambda df: df['Pclass'] < 3 (只解释头/二等舱乘客)
    model_explain(
        train_df, val_df, test_df,
        feature_cols=feature_cols,
        label_col=label_col,
        save_root='data/titanic/shap/',
        categorical_cols=['Sex', 'Embarked'],
        te_smoothing=10,
    )
