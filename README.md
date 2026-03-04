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

- **improved_anchor_explain.py**：补全全局规则学习流程（原代码被注释未启用）；启用反事实解释；修复文件写入模式、编码、零样本除零等 bug；适配 LightGBM 新版 API；用 `optbinning.MDLP` 替换自研离散化（修复排序后标签未对齐的 bug）；缺失值改为中位数/众数填充；自动过滤低覆盖率规则
- **shap_explain.py**：重构为接收原始 DataFrame，直接在连续特征上训练 LightGBM 并生成 SHAP 解释；替换废弃 API（`time.clock`、`early_stopping_rounds`）；图表自动保存
- **plot_helper.py**：修复索引 typo 和类型判断错误；增加 `max_samples` 限制；`bins` 和 `vmax` 改为动态计算以适配小数据集和离散特征
- **anchors/anchor_base.py**：修复 `np.array` 缺少 `dtype=object` 导致的异常
- **其他**：添加 Titanic 示例数据集；移除未使用的 `docs/`；添加 `.gitignore`

## 快速开始

Python 3.11

```bash
pip install shap lightgbm pathos joblib pandas numpy scikit-learn matplotlib dill optbinning

cd src
python improved_anchor_explain.py   # → data/titanic/anchor/result/
python shap_explain.py              # → data/titanic/shap/
```
