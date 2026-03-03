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
│   └── ent_mdlp.py                  # MDLP 离散化
├── data/titanic/                    # Titanic 示例数据集
└── model/                           # 模型文件（自动生成）
```

## 相对原作者的主要改动

- **improved_anchor_explain.py**：补全全局规则学习流程（原代码被注释未启用）；启用反事实解释；修复文件写入模式、编码、零样本除零等 bug；适配 LightGBM 新版 API
- **shap_explain.py**：重构为单文件输入；替换废弃 API（`time.clock`、`early_stopping_rounds`）；图表自动保存
- **plot_helper.py**：修复索引 typo 和类型判断错误；增加 `max_samples` 限制
- **其他**：添加 Titanic 示例数据集；移除未使用的 `docs/`；添加 `.gitignore`

## 快速开始

Python 3.11

```bash
pip install shap lightgbm pathos joblib pandas numpy scikit-learn matplotlib dill

cd src
python improved_anchor_explain.py   # → data/titanic/anchor/result/
python shap_explain.py              # → data/titanic/shap/
```
