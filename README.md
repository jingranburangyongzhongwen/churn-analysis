# churn-analysis

基于论文 *"A Data-Driven Decision Support Framework for Player Churn Analysis in Online Game"* 的流失分析与可解释性框架，使用 **Improved Anchor** 和 **TreeSHAP** 对预测模型进行局部与全局解释。

## 项目结构

```
├── src/
│   ├── improved_anchor_explain.py   # Improved Anchor（MDLP + 多进程 + 反事实 + 全局规则学习）
│   ├── anchor_explain.py            # 原始 Anchor 基线
│   ├── shap_explain.py              # TreeSHAP 特征重要性分析与可视化
│   ├── plot_helper.py               # SHAP 可视化工具
│   ├── anchors/                     # Improved Anchor 核心库
│   └── ent_mdlp.py                  # MDLP 离散化（虽然修复了bug，但已弃用，保留作参考）
├── data/titanic/                    # Titanic 示例数据集
└── model/                           # 模型文件（自动生成）
```

## 相对原作者的主要改动

### `improved_anchor_explain.py`
- 补全全局规则学习流程（原代码被注释未启用），启用反事实解释
- 用 `optbinning.MDLP` 替换自研离散化，修复排序后标签未对齐的 bug
- 缺失值改为中位数 / 众数填充，自动过滤低覆盖率规则
- 适配 LightGBM 新版 API，修复文件写入模式、编码、零样本除零等 bug

### `shap_explain.py`
- 重构为 `train / val / test` 三段式接口
- 修复两处数据泄漏：`LabelEncoder` 全量 fit、early stopping 用 test 集
- 类别特征改用平滑 target encoding（仅在 train 上 fit），单列数值、无共线性、兼容 interventional SHAP
- joblib 并行 TreeSHAP，新增 `shap_filter` / `shap_sample_n` / `top_n_dependence` 控制采样与产图数量
- 产物：`feature_importance.csv`（mean_abs / mean / std / pos_ratio / neg_ratio / 累计占比）+ summary / dependence / force / group 全套图
- 替换废弃 API（`time.clock`、`early_stopping_rounds`）

### `plot_helper.py`
- 修复索引 typo 和类型判断错误
- `bins` 和 `vmax` 改为动态计算，适配小数据集和离散特征
- 增加 `max_samples` 限制

### `anchors/anchor_base.py`
- 修复 `np.array` 缺少 `dtype=object` 导致的异常

### 其他
- 添加 Titanic 示例数据集
- 移除未使用的 `docs/`，添加 `.gitignore`

## 快速开始

Python 3.11

```bash
pip install shap lightgbm pathos joblib pandas numpy scikit-learn matplotlib dill optbinning

cd src
python improved_anchor_explain.py   # → data/titanic/anchor/result/
python shap_explain.py              # → data/titanic/shap/
```
