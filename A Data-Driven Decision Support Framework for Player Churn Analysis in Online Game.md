# A Data-Driven Decision Support Framework for Player Churn Analysis in Online Games

**Published:** KDD '23: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, August 6–10, 2023, Long Beach, CA, USA

**DOI:** https://doi.org/10.1145/3580305.3599759

**ISBN:** 9798400701030

---

## Authors

| Name | Affiliation | Email |
|------|-------------|-------|
| Yu Xiong | Fuxi AI Lab, NetEase Inc., Hangzhou, China | xiongyu1@corp.netease.com |
| Runze Wu* | Fuxi AI Lab, NetEase Inc., Hangzhou, China | wurunze1@corp.netease.com |
| Shiwei Zhao | Fuxi AI Lab, NetEase Inc., Hangzhou, China | zhaoshiwei@corp.netease.com |
| Jianrong Tao | Fuxi AI Lab, NetEase Inc., Hangzhou, China | hztaojianrong@corp.netease.com |
| Xudong Shen | Fuxi AI Lab, NetEase Inc., Hangzhou, China | hzshenxudong@corp.netease.com |
| Tangjie Lyu | Fuxi AI Lab, NetEase Inc., Hangzhou, China | hzlvtangjie@corp.netease.com |
| Changjie Fan | Fuxi AI Lab, NetEase Inc., Hangzhou, China | fanchangjie@corp.netease.com |
| Peng Cui | Tsinghua University, Beijing, China | cuip@tsinghua.edu.cn |

*Corresponding author

**Open Access Support provided by:** NetEase Fuxi Lab, Tsinghua University

---

## CCS Concepts

- **Information systems** → Online analytical processing; Data mining; Enterprise applications.

## Keywords

Churn analysis, Explainable AI, Churn prediction, Anchor, TreeSHAP, A/B evaluation

---

## ACM Reference Format

Yu Xiong, Runze Wu*, Shiwei Zhao, Jianrong Tao, Xudong Shen, Tangjie Lyu, Changjie Fan, and Peng Cui. 2023. A Data-Driven Decision Support Framework for Player Churn Analysis in Online Games. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*, August 6–10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3580305.3599759

---

## Abstract

Faced with saturated market and fierce competition of online games, it is of great value to analyze the causes of the player churn for improving the game product, maintaining the player retention. Numerous research efforts on churn analysis have been made into churn prediction, which can achieve a sound accuracy benefiting from the booming of AI technologies. However, game publishers are usually unable to apply high-accuracy prediction methods in practice for preventing or relieving the churn due to the lack of the specific decision support (e.g., *why they leave and what to do next*). In this study, we fully exploit the expertise in online games and propose a comprehensive data-driven decision support framework for addressing game player churn. We first define the churn analysis in online games from a commercial perspective and elaborate the core demands of game publishers for churn analysis. Then we employ and improve the cutting-edge eXplainable AI (XAI) methods to predict player churn and analyze the potential churn causes. The possible churn causes can finally guide game publishers to make specific decisions of revision or intervention in our designed procedure. We demonstrate the effectiveness and high practical value of the framework by conducting extensive experiments on a real-world large-scale online game, *Justice PC*¹. The whole decision support framework, bringing interesting and valuable insights, also receives positive reviews from the game product teams. Notably, the whole pipeline is readily transplanted to other online systems for decision support to address similar issues.

> ¹ https://leihuo.163.com/en/games/html?g=game-1#nsh

---

## 1 Introduction

In 2022, the growing global game market revenue reaches $184.4 billion, engaging more than 3 billion game players in the world². As an important means of enriching human spiritual life, game has become one of the dispensable ways of entertainment in our daily life. Maintaining sufficient players can bring direct or indirect revenue, which largely determines the survival of online games, especially when competing with the increasing number of similar products released each year. One solution is to constantly attract new players, which is more and more costly and ineffective with the diminishing marginal utility compared with another strategy: addressing the leaving players.

Player churn refers to players leaving the game temporarily or permanently, which reduces the use of the product and is therefore the counterpart of retention [45]. The initial research on user churn is with Customer Relationship Management (CRM), widely adopted in business of Telecom, financial and online game companies and so on [1, 12]. Churn analysis plays a key role in mining players' retention strategies. Some researchers of churn analysis focus on churn prediction [15, 23, 32], trying to identify players' churn behavior in advance through various indicators about the churn (such as the churn probability). However, these accuracy-orientated prediction methods are of little practical value since very few specific decision support can be utilized for preventing or relieving the churn such as *why they leave and what to do next*. Another straightforward way to churn analysis is statistical data analysis, which aims to find valuable insights into user behavior and mindsets, and achieves good performance in traditional scenarios [17]. Nevertheless, the case-by-case schema requiring great human efforts and being difficult to expand, becomes inevitable bottlenecks on nowadays ubiquitous online games where the data are gathered on an unprecedented scale. How to combine the efficiency of the former with the practicality of the latter? How to efficiently get the important information of player churn from massive player data, find out the churn reasons to improve the game, and implement targeted intervention, are the real concerns of game publishers.

To effectively mine potential churn causes, i.e., *why they leave*, we seek inspirations from artificial intelligence methods of more explainable power. One straightforward idea is causal inference, which has been widely concerned in recent years [14, 48]. It seems a good idea to discover the causes for player churn through causality instead of correlation. However, computational causality methods are still in their infancy, learning causal structures from data is only doable in rather limited situations [33]. Different causal effect evaluation algorithms can easily get different or even contradictory results under different settings. Today's plentiful data are more mysterious and make it harder to model responsibly. Another promising path is eXplainable AI (XAI), which is considered as the sine qua non for AI to continue making steady progress. In addition to its important application value in healthcare [5], finance [31] and transportation [16], it also has great potential in online games since we desire the ability to explain black-box models in understandable terms to a human [44]. Moreover, frontier XAI technologies satisfy the ideal properties for churn analysis as shown in Table 1, it is an encouraging idea to adopt XAI into churn analysis in online games for further decision support.

In this paper, we systematically study the problem of churn analysis in online games. We compare several potential churn analysis methods, and propose a decision support framework based on improved XAI technologies to generate potential churn causes for guiding decisions of revision or intervention in online games. The contributions of our study are summarized as follows:

1. We make the first whole XAI-based decision support framework deployed in practical online games. More than traditional prediction and analysis, we highlight the importance of reasoning and intervention for actionable decision support via a comprehensive study of churn analysis methods.
2. We employ a series of acceleration strategies to enhance the performance of the widely cited Anchor tool [36].
3. We provide a new evaluation protocol by collecting user feedback and conducting A/B testing with intervention, which is one of the most critical challenges for XAI methods.
4. We propose a new feature design principle based on churn reasoning experience within MMORPGs. Given the complexity of online games, this feature design principle can be readily transferred to other games or online user services.

> ² https://newzoo.com/insights/articles/the-games-market-in-2022-the-year-in-numbers

---

## 2 Preliminaries

Churn analysis is a crucial aspect of online gaming that seeks to investigate the underlying factors that lead to player attrition, with the ultimate goals of enhancing game quality and extending the game's life cycle. Based on whether players have departed from the game, churn analysis can be categorized into two distinct types: churn player analysis and churn prediction analysis.

### 2.1 Churn Player Analysis

The churn player analysis focuses on understanding the factors that contribute to the churn of players who played daily suddenly stopped. Several prevalent analytical methods include:

- **Statistical data analysis** identifies the common characteristics of churn player based on game data, reconstructs the in-game context, and captures players' psychological state prior to churn. Solutions are then formulated based on objective facts and business insights. For instance, analyze the distribution of tasks, dungeons and maps before the churn.
- **Clustering analysis** employs clustering algorithms to identify specific churn groups based on the churn features [47].
- **Causal analysis** investigates the causal influence of variables on player churn using observational data. The selection bias in observational data is mitigated by eliminating confounding factors that impact both treatment assignment and outcome. Re-weighting and matching-based causal approaches can be employed to evaluate the causal effect of variables on player churn [48].
- **XAI analysis** involves constructing an interpreter to explain a well-trained churn prediction model using XAI technology. It mines significant features and rules impacting churn, often by analyzing the feature importance of logistic regression or decision tree models [15, 21, 32].

Traditional statistical data analysis can yield objective statistical facts, serving as a direct method of analysis. However, it demands high human costs and extensive expertise in the game domain, which impedes scalability. Clustering analysis is susceptible to outliers and lacks appropriate measures to determine feature weighting, thus complicating the attainment of satisfactory outcomes in high-dimensional data. The field of causal inference remains in its nascent stage, necessitating several strong assumptions to be satisfied and posing numerous limitations to various algorithms (e.g., capacity to assess individual effects and continuous variable effects). Crucially, the derived causal effect values lack measures of reliability. In contrast, the field of XAI has witnessed significant advancements in recent years. A number of excellent methods, backed by solid theoretical foundations and statistical evidence, have emerged, demonstrating their practical applicability in analytical contexts. By appropriately leveraging these state-of-the-art algorithms, we can automatically uncover crucial information and deep patterns in the game churn problem. This, in turn, can greatly facilitate our understanding of the underlying reasons for churn, enhance the quality of online games, and ultimately enable the re-engagement of churn players.

### 2.2 Churn Prediction Analysis

Churn prediction analysis aims to forecast whether active players are approaching the decline or exit stages of a game and to identify the contributing factors to this decline. This predictive approach allows for more accessible in-game interventions compared to post-treatment churn player analysis. The two primary methods of churn prediction analysis are as follows:

- **Churn prediction.** The majority of churn analysis research has focused on this method, which includes early-stage churn prediction [4, 29], predicting churn for high-value players with high in-game purchases [23, 38], etc.
- **Churn prediction and explanation.** This method utilizes XAI technology to provide insights into the churn model's predictions and identify key factors influencing individual players' churn.

Solely relying on churn prediction can lead to unsatisfactory intervention outcomes due to the absence of clear explanations for the churn [38]. However, incorporating XAI technology to explain churn predictions equips game developers with crucial information, which can effectively enhance the efficacy of targeted interventions.

Based on the above discussion we conclude that an ideal churn analysis method must be:

- **Scalable (S)**: Easy to adjust for adding new features or analyzing another game.
- **Reliability-Modelable (RM)**: Have objective indicators to support the output analysis results.
- **Individual-Analyzable (IA)**: Able to effectively analyze the churn of individual player.
- **Continuous variable-Analyzable (CA)**: Able to effectively analyze the effect of continuous variable.
- **Predictable (P)**: Able to predict and analyze the churn of players who will be lost.
- **Debiased (D)**: Able to alleviate the spurious effect caused by selection bias in data.

We compare the churn analysis methods according to whether they satisfy the above mentioned properties, and only the XAI analysis has almost all the properties, see Table 1.

### Table 1: The properties of main churn analysis methods

Note: Only the proposed XAI-based method has almost all the required properties.
**S**: Scalable, **RM**: Reliability-Modelable, **IA**: Individual-Analyzable, **CA**: Continuous variable-Analyzable, **P**: Predictable, **D**: Debiased
(✓ indicates presence of property, □ indicates limited presence of property)

| Method | S | RM | IA | CA | P | D |
|--------|---|----|----|-----|---|---|
| Statistical data analysis | | ✓ | | □ | | |
| Clustering analysis | ✓ | ✓ | | □ | | |
| Churn prediction | ✓ | | □ | | ✓ | |
| Causal analysis | ✓ | | □ | □ | | ✓ |
| XAI analysis | ✓ | ✓ | ✓ | ✓ | ✓ | □ |

Game publishers eager to know the churn causes of players; however, these causes are complex and multifaceted, involving both primary and secondary causes, and the interplay of these cause variables further complicates the matter. Additionally, philosophical causality remains unresolved, and causality in the micro-world is not yet clear, making it challenging to pinpoint deterministic causes for player churn. Consequently, we define the data-driven churn analysis in online game from a practical perspective:

> **Definition 2.1 (Data-Driven Churn Analysis in Online Game).** Mining significant objective facts and deep patterns of player churn from game data and integrating them with expert knowledge to establish reliable churn causes. Subsequently, implementing targeted player interventions and game improvements based on the identified causes to mitigate player churn.

---

## 3 Cause Features

The presence of more informative features makes the relationship between features and churn behavior easier for ML model to learn. Furthermore, when these features possess greater relevance to a specific audience, comprehension of the churn is facilitated. In this section, we expound upon the principles underlying the design of churn cause features and, drawing upon our practical experience, design a set of highly referential churn cause features.

### 3.1 Design Principle

Obtaining and quantifying external churn cause information is often challenging; therefore, we concentrate on in-game data mining. The feature design process has predominantly relied on expert experience and lacked systematic guidelines. In this study, we present the first comprehensive set of principles for designing high-quality churn cause feature sets.

- **Well-expressed**: The feature set should comprise features that best express the potential churn causes and indicators that the game developer focuses on (Considering that the game expert knowledge may be imperfect and some churn causes are challenging to express with features).
- **Comprehensive**: To minimize the endogenous problem and enhance churn analysis accuracy, the designed feature set should cover as many churn causes as possible.
- **Actionable**: The designed features should be as actionable as possible, whether through direct or indirect means.
- **Categorical**: Employing categorical features is preferable, as they exhibit a stronger connection to the game domain, which aids in understanding the analysis results. Experts can segment continuous features based on domain knowledge.

### 3.2 Feature Examples

Table 9 presents an exemplary set of common churn cause features as a reference for game operators and designers, enabling them to efficiently construct a high-quality feature set. By leveraging their domain knowledge and adhering to the feature design principles outlined in Section 3.1, they can develop a more tailored, comprehensive, and specific churn cause feature set for their game.

---

## 4 Proposed Framework

Distinct from other studies on player churn analysis, the existing literature primarily focuses on intervention after churn prediction [29, 38] and analysis after churn prediction [15, 21, 32]. In contrast, our proposed framework for churn analysis encompasses churn prediction, explanation, analysis, and a final evaluation and intervention stage, thereby establishing a standard and solid decision support pipeline for player churn analysis, as depicted in Figure 1. The framework comprises six integral modules, including problem setup, feature design and development, model prediction, model interpretation, explanation analysis, and evaluation and intervention. We detail the specific functions of each module as follows.

### Figure 1: The proposed framework for churn analysis.

```
(a) Churn prediction and intervention:
  Data → Model Prediction → Intervention

(b) Churn prediction and analysis:
  Data → Model Prediction → Churn Analysis

(c) Proposed decision support churn analysis framework:
  Data → Model Prediction → Model Interpretation
    ↓                                ↓
  Evaluation & Intervention ← Churn Analysis
```

### 4.1 Problem Setup

Prior to conducting the analysis, it is essential to identify the primary churn issues within the online gaming context through consultations with game operators. Several key questions can guide this process, including:

1. **How should player churn be defined?** Unlike in telecommunication services, where user churn can be determined by subscription cancellations, online gamers seldom delete their accounts even if they have no intention of continuing to play. As such, player churn is typically assessed based on the interval between login and logout events, such as next-day, three-day, and seven-day churn periods.
2. **How can the target players for churn analysis be narrowed down?** Not all players contribute equally to a game company's profits, and some may even negatively impact game services, such as the cheater players. Consequently, high-value, high-level, and new players in newly launched games should be the primary focus of churn analysis.

### 4.2 Feature Design and Development

The development of churn cause features can be effectively and accurately achieved by adhering to the design principles and referring examples as outlined in Section 3. Upon establishing the list of features and their respective calculations, we extract fundamental portraits from the game log, subsequently constructing the cause features centered on these portraits.

### 4.3 Model Prediction

Tree-based ensemble methods, such as random forests and gradient boosted trees, consistently achieve state-of-the-art performance across various domains. These methods have a long-standing history, with recent high-performance implementations becoming an active research area [7, 20, 34]. Although deep learning models have demonstrated remarkable success in image recognition, speech recognition, and natural language processing, tree-based models tend to outperform standard deep models on tabular-form datasets, because features in tabular form have individual meaning and do not have strong multi-scale temporal or spatial structures.

The churn cause features have complex non-linear interactions, necessitating a model with significant flexibility to accurately capture this information. So we opt for gradient-boosting machines, a type of non-parametric model that establishes a parallel between boosting and gradient descent in function space. Fitting trees on large datasets can be computationally challenging; thus, we employ LightGBM [20], a distributed gradient boosting framework designed for high efficiency, faster training speed, lower memory usage, and improved accuracy, for churn model training and prediction.

### 4.4 Model Interpretation

Understanding why a statistical model has made a specific prediction is a key challenge in ML. Many complex models with excellent accuracy, such as gradient boosting, make predictions that even experts struggle to interpret. This forces a trade-off between accuracy and interpretability. In response to this, we employ two sophisticated XAI techniques, TreeSHAP [26] and improved Anchor, to improve the interpretability of the prediction model.

#### 4.4.1 Basic Explainers

**TreeSHAP** is an explanation method for trees that enables the tractable computation of the optimal local explanations called SHAP values, as defined by desirable properties from the classic game-theoretic Shapley values. Feature importance is defined as the change in the expected value of the model's output when a feature is observed versus unobserved. The Shapley values φᵢ(f, x), explaining a prediction f(x), are an allocation of credit among the various features in x. Given a specific prediction f(x), we can compute the Shapley values by using a weighted sum that represents the impact of each feature being added to the model averaged over all possible orders of features being introduced:

$$\phi_i(f, x) = \sum_{S \subseteq S_{all \setminus \{i\}}} \frac{|S|!(N - |S| - 1)!}{N!} [f_x(S \cup i) - f_x(S)]$$

$$= \sum_{S \subseteq S_{all \setminus \{i\}}} \frac{1}{(N \text{choose} |S|)(N - |S|)} [f_x(S \cup i) - f_x(S)] \quad (1)$$

where S represents the subset of all N input features. We utilize the marginal expectation E[f(x), x_S], rather than the conditional expectation E[f(x) | x_S], where x_S denotes a subset of the input vector containing only the features within the set S. As per the principle of causal inference, the "interventional" approach disrupts the dependencies between features [18].

**Anchor** [36] explains individual predictions of any black-box classification model by identifying a decision rule that sufficiently "anchors" the prediction. A rule anchors a prediction if alterations in other feature values do not impact the model's prediction. Anchor addresses a critical limitation of local explanation methods such as LIME [35], which proxy the local behavior of the model in a linear manner. This is insufficient as both the model and the data can exhibit non-linear behavior near the instance. Anchor overcomes this issue by incorporating coverage, the region where the explanation is applicable, into the optimization problem. An anchor A is formally defined as follows:

$$E_{D_x(z|A)} [1_{f(z) = f(x)}] \geq \tau, A(x) = 1 \quad (2)$$

where x represents the instance being explained, A is a set of predicates, i.e., the resulting rule or anchor, such that A(x) = 1 when all feature predicates defined by A correspond to the feature values of x. f denotes the classification model to be explained. It can be queried to predict a label for x and its perturbations. D_x(·|A) indicates the distribution of x's neighbors, matching A. 0 ≤ τ ≤ 1 specifies a precision threshold. Only rule that attain a local fidelity of at least τ are deemed a valid result. Anchor aims to find the rule with the highest coverage among all the rules that meet the given precision threshold. Rules with greater coverage are deemed more significant, as they encapsulate a larger portion of the model. Anchor ensures that the model's decision-making behavior is accurately and effectively conveyed to the user, facilitating a correct understanding and high-fidelity interpretation.

#### 4.4.2 Improved Anchor

Anchor necessitates numerous calls to the ML model for interpreting each instance. Although it employs Multi-Arm Bandit algorithms (MABs) [19] to minimize the number of calls, the computation remains significantly time-consuming, particularly for high-dimensional input data. This constraint hampers its practical applicability. To improve the efficiency and effectiveness of rule generation, we optimize Anchor by implementing three improvement measures: *intermediate variable reuse*, *multiple process* and *discretization in advance*.

**Intermediate Variable Reuse.** When Anchor generates candidate rules using MABs for numerous instances requiring interpretation, it fails to reuse previously calculated precision and sampling times of these candidate rules. This results in considerable repetition of calculations. To address this issue, we store the precision and sampling times of newly encountered candidate rules in memory and retrieve them when subsequent calculations for these rules are required.

**Multiple Process.** Anchor does not rely on interdependence among instances when generating explanation rules. Consequently, we employ multi-process technology to facilitate parallel rule generation, thereby capitalizing on the capabilities of multi-core processors to expedite the process. Furthermore, we implement shared variable technology to enable the exchange of precision and sample times of candidate rules stored in memory, allowing for the efficient reuse of intermediate variables across all processes.

**Discretization in Advance.** In each searching round, Anchor incorporates a feature predicate into the candidate rule. For continuous features, Anchor employs a discretization method based on quantiles (e.g., quartiles or deciles). Consequently, any quantile of continuous features is treated as a binary categorical feature, with values set to 1 for feature values greater than the quantile, and 0 otherwise. This approach substantially expands the search space, resulting in slower search speeds for candidate rules. Furthermore, the derived rule does not easily form a middle form (i.e., value1 < featureName ≤ value2). To address these issues, we apply the Minimum Description Length Principle (MDLP) [9] for discretizing continuous features in advance. Unlike traditional entropy-based discretization methods, MDLP concentrates on data compression principles to return the minimal number of optimal cut points. The improved approach of discretizing continuous features in advance and interpreting them as categorical features ensures that the search space size equals the feature dimension. This not only considerably accelerates rule generation and enhances the effectiveness of explanation, but also improve the accuracy of the churn prediction model.

### 4.5 Explanation Analysis

In our study, we utilize four distinct types of applications, drawing upon the theoretical underpinnings of TreeSHAP and the improved Anchor, to systematically examine the underlying causes contributing to player churn. These applications encompass individual churn explanation, group churn explanation, global churn explanation and churn feature dependence explanation.

#### 4.5.1 Individual Churn Explanation

Both TreeSHAP and the improved Anchor generate local explanations. To provide a more intuitive visual analysis, we utilize force plots for displaying SHAP values. Figure 2 demonstrates each feature's contribution in pushing the model output from its base value (the average model output over the training dataset) to the final model output. Features that increase the churn prediction are depicted in red, while those that decrease it are in blue.

**Figure 2: Force plot for individual churn explanation.**
The force plot shows output value = 0.97 (base value is the average model output). The four labeled features and their values for this player are:
- Total times of encounter tasks completed = 130 (pushes toward churn, red)
- Percentage of churn friends = 0.6 (pushes toward churn, red)
- Guild fund = -1 (pushes toward churn, red)
- Number of advanced skill book learning = 23 (pushes away from churn, blue)

The interpretability of the rules is intuitive, easily comprehensible, and demands minimal effort to draw conclusions. For explanation rules generated by the improved Anchor, excessive predicates may diminish coverage and human readability while enhancing precision. Given that human short-term memory can simultaneously process approximately 5 to 9 items [28], we restrict the length of a single rule to 5 in practice. We present a rule explanation instance for the same player in a textual format, as illustrated in Table 7a. These two explanations can corroborate and supplement each other.

#### 4.5.2 Group Churn Explanation

Distinct from unsupervised clustering, which relies on feature values, supervised clustering is based on SHAP values. By utilizing feature importance, supervised clustering enables the natural conversion of all input features into values with consistent units, addressing the challenge of determining feature weighting in unsupervised clustering. Additionally, supervised clustering mitigates the impact of outliers. Figure 6 illustrates the application of supervised clustering for group churn explanation.

Each local explanation rule is accompanied by quantifiable accuracy, coverage, and the number of players covered. The resulting churn group not only provides an intuitive description of churn explanation but also offers significant objective statistical support. Local rules are sorted in descending order based on accuracy, with a focus on rules with accuracy greater than the rate of natural churn and coverage of more than 100 players, see Table 2.

**Figure 6: Supervised clustering for group churn explanations.**
(The figure shows a scatter plot of samples ordered by similarity on the x-axis and SHAP value on the y-axis, with two identified churn clusters highlighted — churn cluster 1 and churn cluster 2 — shown in pink/red.)

#### 4.5.3 Global Churn Explanation

The global churn explanation offers a comprehensive understanding of the model, facilitating a rapid grasp of the underlying causes of churn. In the case of TreeSHAP, to obtain an overview of the features with the most significant impact on player churn, we plot the SHAP values for every feature across all samples. Figure 3 presents this summary plot, which organizes features based on the cumulative magnitude of their SHAP values and uses these values to depict the distribution of each feature's influence. The color indicates the feature value, with red denoting high and blue signifying low values.

For the improved Anchor, we adopt a submodular-pick (SP) algorithm that identifies an optimal subset of rules for conveying the global behavior of the churn model. Rather than randomly selecting rules, the SP approach selects K rules that encompass the maximal number of instances within the dataset.

**Figure 3: Summary plot for global churn explanations.**
Features ranked by cumulative SHAP magnitude (top to bottom, most to least important):
1. Titles
2. Advanced skill books
3. Unbound Yuanbao acquisition
4. Tasks abandoned
5. Kicked out of team
6. Experience acquisition
7. Total score improved
8. Friends
9. Guild fund
10. Copper money obtained
11. Level promotion
12. Plot watching
13. Mainline tasks completed
14. Churn friends
15. Skill experience acquisition
16. Guild tasks completed
17. Branch tasks completed
18. YX common tasks completed
19. Death in leisure tasks
20. SHL tasks completed

Color indicates feature value (red = high, blue = low); x-axis is SHAP value (impact on model output).

#### 4.5.4 Churn Feature Dependence Explanation

The Churn feature dependence plot uses the SHAP value of a specific churn feature as the y-axis, while the corresponding feature value is represented on the x-axis. By plotting these values for all individuals within the dataset, we can observe how the attributed importance of the feature fluctuates in relation to churn as the feature value varies, see Figure 4. Additionally, color is employed to signify player density, with darker shades indicating a higher concentration of players.

**Figure 4: Churn feature dependence plot for plot watching.**
X-axis: Feature value (Percentage of plot watching, 0–100%). Y-axis: SHAP value (impact on model output, range approximately −1.5 to +0.5). Color represents player density (darker/warmer = higher concentration). Key pattern: SHAP value is positive (increases churn) roughly between 60%–95%; negative (reduces churn) below ~60% and above ~95%.

### 4.6 Evaluation and Intervention

#### 4.6.1 User Study

Game operators conduct user research on the churn players to get feedback on the reasons for their churn. The forms include telephone interviews and questionnaires. Based on the feedback, the game operators implement targeted, small-scale interventions to improve the return likelihood of high-value players. Additionally, the identified cause features are refined and iterated based on the discrepancies observed between the feedback and the generated explanations.

#### 4.6.2 Intervention

Intervention serves as an important means to explore the commercial value of the decision support churn analysis framework. By integrating the explanations and user study to summarize the clear churn causes, game operators or developers can deploy targeted interventions. Moreover, we employ A/B testing to assess the efficacy of the interventions in a systematic manner.

---

## 5 Churn Analysis and Evaluation

### 5.1 Dataset

We utilize four datasets, comprising a real game dataset for churn analysis and evaluation, as well as three public real datasets to evaluate the efficiency of the improved Anchor.

The real game dataset employed for the experiments is sourced from Justice Online, a real-world MMORPG in NetEase. This dataset encompasses log data from high-value players over a four-month period in 2020, containing 75,680 churn players and 136,523 retained players based on the 7-day churn definition. We have devised 82-dimensional churn cause features, such as the number of friends, guild fund, the number of titles. A detailed description of these typical cause features can be found in Table 10.

The three public real datasets consist of: (i) *Titanic*, comprising 891 passenger instances, with the objective of predicting a passenger's survival based on 7 features such as sex, age, class, etc.; (ii) *Adult*, containing 48,842 instances and 15 demographic features like age, workclass, martial-status, race, etc., with the goal of predicting whether an individual earns more than 50k USD annually; and (iii) *German*, in which each of the 1,000 entries is classified as a "good" or "bad" creditor according to 20 features including age, sex, credit amount, purpose, etc. The codes for reproducing the key results can be accessed online via the provided link¹.

> ¹ https://github.com/fuxiAllab/churn-analysis

### 5.2 Churn Analysis

#### 5.2.1 Individual Analysis

The player depicted in Figure 2 demonstrates the guild fund (with a value -1 means that the player is not in a guild), churn friends rate, and the total number of Qiyu gameplay completions have significant effects on the churn, indicating that the player churn is primarily influenced by social factors. The rule explanation presented in Table 7a further highlights that the guild-related issues and task abandonment are critical causes of churn. Moreover, several other factors, such as the number of completed mainline tasks, level promotion, and game plot watching, suggest that the player encounter difficulties in accomplishing mainline tasks.

#### 5.2.2 Group Analysis

Figure 6 illustrates the presence of two distinct churn clusters identified through supervised clustering. The primary cause for the churn in Cluster 1 is attributed to difficulties experienced in upgrading their equipment and skills. Conversely, the primary cause in Cluster 2 is linked to social influence factors, such as friendship and guild are in a bad state. The group churn rules presented in Table 2 reveal two churn groups who prefer to skip the game plot. While one group's churn is primarily driven by social influence, the other is due to the inability to advance their skill level.

#### 5.2.3 Global Analysis

Figure 3 displays the results, indicating title acquisition, advanced skill book learning, unbound Yuanbao acquisition, task abandoned, team expulsion and plot watching are the most important factors contributing to the churn. In the game, players obtain unique titles by completing specific challenging or engaging events. Acquiring these titles imparts a strong sense of achievement, especially for rarer titles. As the most advanced equipment is relatively easy to obtain in-game, skill development becomes crucial for character improvement. Unbound Yuanbao, acquired through in-game recharging, can be utilized to purchase time cards, fashions, and various advanced virtual items. Consequently, players with higher recharging tend to have better gaming experiences and retention. An interesting issue regarding game plot watching was discovered and explored in a detailed experiment outlined in Section 5.3.4. The global rule explanations presented in Table 8 also suggest that acquiring sufficient titles is a key factor in player retention and reveal additional important churn factors, such as friends churning, not joining a healthy guild, and a low number of completed mainline tasks.

#### 5.2.4 Dependence Analysis

Figure 4 illustrates the influence of the percentage of game plot watching on churn behavior at various levels. When this percentage is below 60%, it positively affects player retention. However, when it exceeds 60%, there is a noticeable impact on player churn. A high player density suggests that a significant number of players do not appreciate the game plot. Interestingly, when the percentage surpasses 95%, player retention tends to increase, indicating that a small number of players really enjoy the game plot.

### 5.3 Evaluation

We carried out a series of experiments to assess the efficacy of our decision support churn analysis framework in terms of the following aspects: computational performance, model prediction accuracy, analysis human evaluation and intervention effectiveness.

#### 5.3.1 Computational Performance

Naturally, constructing an interpreter for the model requires additional time. We assess the algorithmic efficiency of TreeSHAP and the improved Anchor. On average, the local churn explanation generated by TreeSHAP takes 0.0062 seconds, while the improved Anchor takes 10.1 seconds on a PC equipped with an Intel Core i7 3.2GHz quad-core CPU and 16GB RAM. In comparison to the quartile discretization Anchor, which requires an average of 484.9 seconds to generate a churn rule, the improved Anchor improves efficiency by more than 40 times. Table 3, illustrates the efficiency improvement of the improved Anchor across all four datasets in comparison to the original Anchor. Additionally, the table highlights improvement contributed by the three individual components, i.e., intermediate variables reuse (IVR), multiple process (MP), and discretization in advance (DIA).

### Table 3: Runtime comparison of Anchor methods and with/without three improvement components

| Method | Titanic | Adult | German | Game |
|--------|---------|-------|--------|------|
| Anchor (quartile) | 0.26s | 1.1s | 0.98s | 484.9s |
| Proposed (w DIA, w/o IVR,&MP.) | 0.22s | 0.58s | 0.37s | 104.6s |
| Proposed (w IVR, w/o DIA,&MP.) | 0.23s | 0.52s | 0.42s | 79.2s |
| Proposed (w IVR,&DIA, w/o MP.) | 0.18s | 0.5s | 0.38s | 30.7s |
| **Proposed (w IVR,&DIA,&MP.)** | **0.05s** | **0.16s** | **0.12s** | **10.1s** |

#### 5.3.2 Model Prediction Accuracy

Only when the model captures the complex internal relationships within the data can the interpretation be meaningful. The accuracy of the model serves as the foundation for obtaining reasonable explanations. In Table 4, we compared the performance of several primary churn analysis models and advanced ML models. Our experiment used 5-fold cross-validation, and the optimal settings of the corresponding models were employed. The results demonstrate that tree-based ensemble models possess a robust ability to handle tabular data, and the LightGBM with MDLP discretization of continuous features, which was implemented in our framework, achieved the highest performance.

### Table 4: Model prediction accuracy (%) of different models

| Method | Accuracy | Precision | Recall | AUC |
|--------|----------|-----------|--------|-----|
| Logistic Regression [21] | 85.50 | 80.37 | 76.01 | 92.02 |
| Decision Tree [15] | 83.46 | 75.80 | 76.15 | 81.82 |
| Survival Forest [32] | 89.26 | 87.21 | 78.79 | 96.18 |
| MLP | 88.48 | 84.05 | 81.30 | 95.64 |
| Random Forest | 85.54 | 83.93 | 70.89 | 92.83 |
| CatBoost | 90.91 | 87.43 | 84.83 | 97.32 |
| XGBoost | 90.90 | 87.49 | 84.83 | 97.53 |
| LightGBM | 91.10 | 87.99 | 84.83 | 97.57 |
| **LightGBM(MDLP)** | **92.21** | **88.83** | **85.10** | **97.76** |

#### 5.3.3 Analysis Human Evaluation

The assessment of consistency can be conducted by comparing player feedback with the results of churn analysis. We have gathered 190 instances of feedback through telephone interviews and questionnaires. The primary causes of churn include: busy, monotonous gameplay, challenges in acquiring equipment, friends churn, issues with team formation and matchmaking, difficulties in obtaining resources, and problems with guild funding activities, among others. Certain feedback can be attributed to external factors, such as busy, while others do not adequately express the underlying causes of churn, such as monotonous gameplay; we have excluded these instances from our analysis. Out of the 116 relevant pieces of feedback, we matched them with corresponding individual explanations in the churn analysis results. More specifically, we considered the top 5 features of churn impact importance ranking and supplemented them with explanation rules. Table 7 illustrates two examples of such matching. Overall, approximately 63% of player feedback aligns with our analysis results.

We compared our matching results with other prominent churn analysis methods. [21] used the weight coefficient of binomial logistic regression to analyze user churn in mobile telecommunication. [15] employed the feature importance of decision trees to examine the reasons behind player churn. [32] applied survival analysis to model player churn and subsequently analyzed churn based on the feature importance of conditional inference survival ensembles. Given that the feature importance generated by these methods is global, we matched player feedback based on the top 5 features of global importance for each method. Table 5 displays the comparison of matching results between these methods and our own; our method yields the best outcomes, thereby highlighting the superiority of our method for churn analysis. Additionally, we compared the capacity of these methods to extract critical information. Figure 5 shows the AUCs of these methods obtained by retraining a LightGBM model based on a varying number of features ranked top in global importance. SHAP demonstrates superior performance in capturing important information.

**Figure 5: Prediction with top K important features.**
X-axis: Number of the most importance features (10–50). Y-axis: AUC (0.90–0.97). Four curves compared: LR (Logistic Regression), Decision Tree, Survival Forest, SHAP. SHAP curve lies consistently highest across all K values, confirming its superior ability to capture the most important features.

### Table 5: Human accuracy (%) comparison of several main churn analysis methods

| Method | Accuracy |
|--------|----------|
| Logistic Regression [21] | 22.41 |
| Decision Tree [32] | 37.07 |
| Survival Forest [32] | 38.79 |
| Anchor | 43.97 |
| Improved Anchor | 47.41 |
| TreeSHAP | 53.45 |
| **Ours** | **62.93** |

#### 5.3.4 Intervention Effectiveness

Churn analysis produces commercial value through the implementation of interventions. Our results indicate that players with a high percentage of plot-watching are prone to churn, a factor seldom mentioned in player feedback. In Justice Online, to familiarize players with game plots, they are only allowed to skip plots after viewing at least 80% of the content. Our analysis suggests that this setting has a discernible effect on player churn, as a lot of plots are not well-executed, and most players prefer not to spend excessive time watching the game plot.

We implemented an A/B evaluation to assess the impact of the setting in terms of online time in recent two weeks. Active players on a specific server were randomly assigned to two groups: the experimental group (28,873 players), who were permitted to skip the plot at any time, and the control group (26,714 players), who maintained the original settings. The 7-day churn rate for the experimental group was 1.6 percentage points lower than that of the control group. We utilized a two independent samples t-test to determine if there were significant differences between the groups. The null hypothesis posited that the average online time in the recent two weeks, at two weeks post-test, was equal for both groups. The test statistic Z was calculated using the t-test formula as follows:

$$Z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \quad (3)$$

where x₁ and x₂ represent the mean values of the observation indicator (i.e. average online time in the recent two weeks) of the experimental and control group, respectively, S denotes the standard deviation of the observation indicator, and n represents the sample size. The p-value corresponding to Z value was calculated using the t-distribution formula.

### Table 6: Two independent samples t-test average online time in the recent two weeks results

| Group | Churn Rate | Mean Time | Z-statistics | P-value |
|-------|------------|-----------|--------------|---------|
| experimental | 27.5% | 18,738s | 3.28 | 0.001 |
| control | 29.1% | 18,392s | | |

The results, displayed in Table 6, show that the average online time for the experimental group was 346 seconds longer than that of the control group. The p-value indicates strong statistical evidence that players activity was positively influenced by our intervention. In other words, the percentage of plot-watching, a potential cause identified by our method, is indeed a precise cause of player churn. Currently, the game developer has implemented a system allowing players to skip the plot at any time on all servers. One month after adopting this scheme, the global importance ranking of this feature dropped from 7th to 26th, further validating our conclusion.

---

## 6 Related Work

### 6.1 Churn Prediction

Churn prediction is a prevalent issue that has been extensively examined in various domains such as Telecom [22], finance [2], online games [15, 23–25, 32, 37, 38, 43, 46], e-commerce [49]. Numerous ML algorithms have been employed in diverse churn prediction settings, including widely used logistic regression [15, 38], tree-based model [38], SVM [30] and neural network [23, 38]. In the context of online gaming, most studies model churn prediction as a binary classification problem [15, 23, 38]. This approach involves using domain knowledge and data analysis to extract user portraits and historical behavior features related to churn as inputs for the ML model, which then predicts whether a player will churn within a specific future time window. In the other part, some studies model churn prediction as the time until churn, employing traditional regression methods or survival analysis techniques that can accommodate censored data [23, 32].

Despite considerable research efforts dedicated to churn prediction, few studies have generated actual commercial value. To the best of our knowledge, our decision support churn analysis framework is the first one that combines churn prediction and advanced XAI methods to produce intuitive churn causes and implement targeted interventions on game industry applications, ultimately bringing tangible commercial value.

### 6.2 Explainable AI

The goal of XAI is to improve the interpretability and reliability of ML models while maintaining the high performance. XAI has rapidly emerged as a burgeoning research area, yielding a plethora of diverse methods in recent years [3]. Depending on whether a model has built-in interpretability, XAI can be categorized into ante-hoc and post-hoc interpretations. Ante-hoc interpretation understand the model without additional information, such as decision tree and linear models. Post-hoc interpretation uses additional techniques to explain the mechanism and decision basis of complex black-box model, encompassing both global and local explanations.

Global explanations focus on understanding the complex logic and internal working mechanism of the model as a whole. It mainly includes rule extraction and model distillation. Rule extraction [8, 27] provides an understanding of the overall decision logic of the complex model by extracting the explanation rules from the trained model. Model distillation [11, 42] aims to compress the functions of complex teacher models into smaller and faster student models with similar performance.

Local explanations focus on understanding the decision-making process and the basis of learning model for particular inputs. It mainly includes sensitivity analysis, local approximation interpretation and back propagation interpretation. Sensitivity analysis [10] is a kind of uncertainty analysis technology to study the impact of some changes of independent variables on dependent variables from the perspective of quantitative analysis. Local approximation interpretation [13, 35, 36] is to use a simple interpretable model to fit the decision results of a certain input case, and explain the decision results based on the interpretable model. Back propagation interpretation [6, 39–41] aims to use the back propagation mechanism of DNN to propagate the important signals of model decision from the output layer neurons of the model layer by layer to the input to derive the feature importance of the input samples.

Many XAI methods generate local explanations; however, challenges persist in terms of explanation accuracy, readability and algorithmic efficiency. In this paper, we present the application of SHAP values [26], a promising technique backed by solid theoretical foundations, and the improved Anchor, which efficiently generates intuitive and comprehensible explanation rules with clear defined coverage, to reason the player churn.

---

## 7 Conclusion

Churn analysis in online games is a complex research area that intersects both engineering and business administration fields, making it challenging for researchers to synthesize or comprehend from a single study. This study offers a comprehensive examination of this intricate yet highly valuable topic, providing guidance and directions for future research on churn. In this paper, we compare the primary churn analysis methods based on several critical properties and provide a practical definition of player churn analysis. We also summarize the design principles of churn cause features and supply a reference feature set. More importantly, we improve Anchor through three key measures, resulting in superior efficiency as well as better effectiveness, and introduce a novel and practical XAI-based decision support analysis framework for player churn. The analysis results yield significant insights into player churn, which are highly valued by game operators and designers, and have generated tangible commercial value in many games. In future work, we will explore churn analysis in two directions: first, by reinforcing the framework's ability to eliminate spurious correlations, and second, by attempting to apply our churn analysis framework to various types of online games.

---

## A Supplement

### A.1 Examples of Features

Table 9 provides a set of players churn cause features for reference, to help game operators and designers efficiently design a set of high-quality features for they own. The feature set contains typical churn cause information including hardware limitations, basic information, art style, novice guide, social influence, gameplay influence, etc. The features in our game dataset are designed with reference to the features in Table 9 and combined with the domain knowledge of the game expert. Table 10 shows typical examples of these features.

### A.2 Hyperparameter Settings

In our experiments, we set learning rate η = 0.05 and 1000 trees in tree-based prediction model. We also use bagging when constructing each tree, where trees are trained on a random 80% sub-sample of the data and a random 90% sub-sample of the columns. The maximum tree depth is 6 and the minimum child weight of any branch is 20. For the improved Anchor, we choose the fidelity threshold τ = 0.95 and the statistical confidence δ = 0.1, limit the length of rule to 5 and set the number of processes to 4.

---

## Tables

### Table 2: Top 2 Group Churn Explanation Rules

| Rule | Length | Accuracy (%) | Numbers |
|------|--------|--------------|---------|
| 0<Titles≤30 & 0<Friends≤6 & 0<Guild fund≤8,750,065 & 80.52<Percentage of plot watching ≤96.64 | 4 | 97.28 | 1880 |
| 0<Titles≤30 & 0<Advanced skill books≤19 & 80.52<Percentage of plot watching ≤96.64 | 3 | 96.49 | 5984 |

### Table 7: Two Examples of Comparison Among Local Explanation (a) and Corresponding Player Feedback (b) (Matching churn causes highlighted in bold)

**(a) Individual churn explanation.**

| Player | Feature | Value | SHAP Values | Rules |
|--------|---------|-------|-------------|-------|
| Player A | **Guild fund** | **-1** | **0.7266** | 0.01 < Percentage of tasks abandoned ≤ 1.00 & **Guild fund = -1.00** & 0.00 < Daily average number of mainline tasks completed ≤ 18.24 & 2.00 < Level promotion ≤ 10.00 & 80.52 < Percentage of plot watching ≤ 96.64 |
| Player A | **Percentage of churn friends** | **0.6** | **0.4057** | |
| Player A | **Total times of Qiyu gameplay completed** | **130** | **0.3970** | |
| Player A | Percentage of tasks abandoned | 0.037 | 0.3176 | |
| Player A | Percentage of plot watching | 83.24 | 0.1992 | |
| Player B | **Percentage of churn friends** | **1** | **0.4931** | 0 < Percentage of experience acquisition ≤ 0.34 & **0 < Number of friends ≤ 3.00** & 0 < Number of titles ≤ 121.00 & 80.52 < Percentage of plot watching ≤ 96.64 & 0.58 < Percentage of teams created ≤ 1.00 |
| Player B | Percentage of time of PVP tasks completed | 0.0282 | 0.3850 | |
| Player B | **Number of friends** | **2** | **0.3616** | |
| Player B | Percentage of plot watching | 81.21 | 0.2572 | |
| Player B | Number of daily re-forging equipment | 0.11 | 0.2311 | |

**(b) Player feedback.**

| Player | Feedback |
|--------|---------|
| Player A | I used to play the game with my friends, but now **most of them have left**. **My guild goes from bad to worse**. By the way, I played **Qiyu gameplay** many times and the gameplay is boring now. These made me disappointed. |
| Player B | There is nothing new when I go up to full level in the game. Moreover, **a couple of my best friends have left**. I feel bored and lonely to continue the game. |

### Table 8: Top 5 Global Churn Rules Selected by SP Algorithm

Rules with an accuracy greater than 90% are selected as candidate rules. The coverage measures the coverage of the current rule set on the churn players in the dataset and the accuracy represents the rule performance of churn classification.

| Rank | Rule | Coverage | Accuracy |
|------|------|----------|----------|
| 1 | 0<Titles≤30 & 0<Advanced skill books≤19 & 0<Friends≤6 | 8.5% | 91.95% |
| 2 | 30<Titles≤57 & 0<Daily average number of mainline tasks completed≤8.24 & 80.52<Percentage of plot watching≤96.64 & 0<Friends≤6 | 13.48% | 92.69% |
| 3 | Daily average number of YX common tasks completed≤0 & 0<Daily average number of mainline tasks completed≤8.24 & 0<Kicked out of team≤3 | 16.22% | 92.43% |
| 4 | 0<Advanced skill books≤19 & 0<Total score improved≤64 & 0<Tasks abandoned≤4 | 19.18% | 93.01% |
| 5 | 0<Titles≤30 & 0<Guild fund≤8,750,065 & 0<Tasks abandoned≤4 & 0<Daily average number of mainline tasks completed≤8.24 | 21.01% | 94.26% |

### Table 9: Churn Cause Feature Examples in Online Games

| Category | Churn Cause | Feature |
|----------|-------------|---------|
| **Limited Hardware** | Limited Device | Device model |
| | Limited Operating System | Operating system version |
| | Limited Channel | App Channel |
| | Server Instability | Server |
| **Basic Information** | Role Class Unfriendliness | Class |
| | Role Race Unfriendliness | Race |
| **Art Style** | Unappreciated Character | Time spent on customizing |
| | Unappreciated Animation | Animation watching percentage |
| **Novice Guide** | Complicated Guide | Guide stop step |
| | Long Time Guide | Time spent on guide |
| **Social Influence** | Few Friends | Unchurn friends number, Churn friends proportion |
| | Few Teams | Team times, Kicked times |
| | Spouse Leaves | Spouse logout days |
| | Master Leaves | Master logout days |
| | Guild Decline | Unchurn guild members proportion, Guild fund |
| **Gameplay Influence** | Difficult Key Level/Task | Stop level/task/map |
| | Gameplay A Influence | Gameplay A participation/failure times |
| **Role Growth** | New Player Unfriendliness | Player level & stop days |
| | Old Player Bored | Player level & stop days |
| | PVP Growth Blocked | Ladder level & stop days |
| | Equipment Growth Blocked | Equipment Score & stop days |
| **Economic System** | Difficult Coin Obtain | Coin acquisition, Coin consumption |

### Table 10: Churn Cause Features in the Dataset

| Category | Typical Feature Examples |
|----------|--------------------------|
| **Art Style** | Time spent on customizing, percentage of plot watching |
| **Social Influence** | The number of days master didn't log in, the number of days apprentice didn't log in, the number of days couple didn't log in, number of friends, proportion of churn friends, guild level |
| **Gameplay Influence** | The number of daily branch tasks completed, the number of daily mainline tasks completed, the number of guild tasks completed, the number of stores owned, amount of funds obtained by stores, number of daily re-forging equipment, proportion of home time used, proportion of related dungeon time used, proportion of related tasks completed, daily average time spent in prison, the number of deaths in related dungeon, proportion of being killed by players, average waiting time of team formation, the number of master-apprentice tasks completed, number of dissolution of master-apprentice relationship, number of stores bankruptcy, number of being kicked out of guild, number of being kicked out of team, task abandon times, the number of deaths in leisure tasks, the number of YX common tasks completed, the number of SHL tasks completed, total times of Qiyu gameplay completed |
| **Role Growth** | Number of level improvement, total role score improvement, number of equipment score improvement, number of titles, number of advanced equipment upgrade, the average number of skill level improvement, number of advanced skill book learning, proportion of experience acquisition, proportion of skill experience acquisition |
| **Economic System** | Guild fund, advanced equipment acquisition quantity, proportion of copper money acquisition quantity, unbound Yuanbao acquisition quantity |

---

## References

[1] Jaehuyn Ahn. 2020. A Survey on Churn Analysis. *ArXiv abs/2010.13119* (2020).

[2] Ozden Gür Ali and Umut Aritürk. 2014. Dynamic churn prediction framework with more effective use of rare event data: The case of private banking. *Expert Systems with Applications* 41, 17 (2014), 7889–7903.

[3] Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, Adrien Bennetot, Siham Tabik, Alberto Barbado, Salvador García, Sergio Gil-López, Daniel Molina, Richard Benjamins, et al. 2020. Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion* 58 (2020), 82–115.

[4] Valerio Bonometti, Charles Ringer, Mark Hall, Alex R Wade, and Anders Drachen. 2019. Modelling early user-game interactions for joint estimation of survival time and churn probability. In *2019 IEEE Conference on Games (CoG)*. IEEE, 1–8.

[5] Zhengping Che, Sanjay Purushotham, Robinder Khemani, and Yan Liu. 2016. Interpretable deep models for ICU outcome prediction. In *AMIA annual symposium proceedings*, Vol. 2016. American Medical Informatics Association, 371.

[6] Hugh Chen, Scott Lundberg, and Su-In Lee. 2021. Explaining models by propagating Shapley values of local components. In *Explainable AI in Healthcare and Medicine*. Springer, 261–270.

[7] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*. ACM, 785–794.

[8] Houtao Deng. 2019. Interpreting tree ensembles with intrees. *International Journal of Data Science and Analytics* 7, 4 (2019), 277–287.

[9] Usama Fayyad and Keki Irani. 1993. Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning. In *13th International Joint Conference on Artificial Intelligence*, Vol. 1022–1027.

[10] Ruth C Fong and Andrea Vedaldi. 2017. Interpretable explanations of black boxes by meaningful perturbation. In *Proceedings of the IEEE International Conference on Computer Vision*. 3429–3437.

[11] Nicholas Frosst and Geoffrey Hinton. 2017. Distilling a neural network into a soft decision tree. *arXiv preprint arXiv:1711.09284* (2017).

[12] David L García, Àngela Nebot, and Alfredo Vellido. 2017. Intelligent data analysis approaches to churn as a business problem: a survey. *Knowledge and Information Systems* 51, 3 (2017), 719–774.

[13] Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Dino Pedreschi, Franco Turini, and Fosca Giannotti. 2018. Local rule-based explanations of black box decision systems. *arXiv preprint arXiv:1805.10820* (2018).

[14] Ruocheng Guo, Lu Cheng, Jundong Li, P Robert Hahn, and Huan Liu. 2020. A survey of learning causality with data: Problems and methods. *ACM Computing Surveys (CSUR)* 53, 4 (2020), 1–37.

[15] Fabian Hadiji, Rafet Sifa, Anders Drachen, Christian Thurau, Kristian Kersting, and Christian Bauckhage. 2014. Predicting player churn in the wild. In *2014 IEEE Conference on Computational Intelligence and Games*. IEEE, 1–8.

[16] Jacob Haspiel, Na Du, Jill Meyerson, Lionel P Robert Jr, Dawn Tilbury, X Jessie Yang, and Anuj K Pradhan. 2018. Explanations and expectations: Trust building in automated vehicles. In *Companion of the 2018 ACM/IEEE International Conference on Human-Robot Interaction*. 119–120.

[17] Shin-Yuan Hung, David C Yen, and Hsiu-Yu Wang. 2006. Applying data mining to telecom churn management. *Expert Systems with Applications* 31, 3 (2006), 515–524.

[18] Dominik Janzing, Lenon Minorics, and Patrick Blöbaum. 2020. Feature relevance quantification in explainable AI: A causal problem. In *International Conference on Artificial Intelligence and Statistics*. PMLR, 2907–2916.

[19] Emilie Kaufmann and Shivaram Kalyanakrishnan. 2013. Information complexity in bandit subset selection. In *Conference on Learning Theory*. 228–251.

[20] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems*. 3146–3154.

[21] Abbas Keramati and Seyed MS Ardabili. 2011. Churn analysis for an Iranian mobile operator. *Telecommunications Policy* 35, 4 (2011), 344–356.

[22] Kyoungok Kim, Chi-Hyuk Jun, and Jaewook Lee. 2014. Improved churn prediction in telecommunication industry by analyzing a large network. *Expert Systems with Applications* 41, 15 (2014), 6575–6584.

[23] EunJo Lee, Yoonjae Jung, Du-Ming Shi, Jaehoon Jeon, Sung-il Yang, SangKwang Lee, Dae-Wook Kim, Pei Pei Chen, Anna Guitart, Paul Bertens, et al. 2018. Game data mining competition on churn prediction and survival analysis using commercial game log data. *IEEE Transactions on Games* (2018).

[24] Jiayu Li, Hongyu Lu, Chenyang Wang, Weizhi Ma, Wan Zhang, Xiangyu Zhao, Wei Qi, Yiqun Liu, and Shaoping Ma. 2021. A Difficulty-Aware Framework for Churn Prediction and Intervention in Games. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*. 943–952.

[25] Xi Luo, Muhe Xie, Xidao Wen, Rui Chen, Yong Ge, Nick Duffield, and Na Wang. 2020. Micro- and macro-level churn analysis of large-scale mobile games. *Knowledge and Information Systems* 62, 4 (2020), 1465–1496.

[26] Scott M Lundberg, Gabriel G Erion, and Su-In Lee. 2018. Consistent individualized feature attribution for tree ensembles. *arXiv preprint arXiv:1802.03888* (2018).

[27] Morteza Mashayekhi and Robin Gras. 2017. Rule extraction from decision trees ensembles: new algorithms based on heuristic search and sparse group lasso methods. *International Journal of Information Technology & Decision Making* 16, 06 (2017), 1707–1727.

[28] George A Miller. 1956. The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological review* 63, 2 (1956), 81.

[29] Miloš Milošević, Nenad Živić, and Igor Andjelković. 2017. Early churn prediction with personalized targeting in mobile social games. *Expert Systems with Applications* 83 (2017), 326–332.

[30] Julie Moeyersoms and David Martens. 2015. Including high-cardinality attributes in predictive models: A case study in churn prediction in the energy sector. *Decision support systems* 72 (2015), 72–81.

[31] Jean Jacques Ohana, Steve Ohana, Eric Benhamou, David Saltiel, and Beatrice Guez. 2021. Explainable AI (XAI) Models Applied to the Multi-Agent Environment of Financial Markets. In *International Workshop on Explainable, Transparent Autonomous Agents and Multi-Agent Systems*. Springer, 189–207.

[32] Africa Periáñez, Alain Saas, Anna Guitart, and Colin Magne. 2016. Churn prediction in mobile social games: Towards a complete assessment using survival ensembles. In *2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)*. IEEE, 564–573.

[33] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. 2017. *Elements of causal inference: foundations and learning algorithms*. The MIT Press.

[34] Laudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems*. 6638–6648.

[35] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. Why should I trust you?: Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining*. ACM, 1135–1144.

[36] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2018. Anchors: High-precision model-agnostic explanations. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 32.

[37] Karsten Rothmeier, Nicolas Pflanzl, Joschka A Hüllmann, and Mike Preuss. 2020. Prediction of player churn and disengagement based on user activity data of a freemium online strategy game. *IEEE Transactions on Games* 13, 1 (2020), 78–88.

[38] Julian Runge, Peng Gao, Florent Garcin, and Boi Faltings. 2014. Churn prediction for high-value players in casual social games. In *2014 IEEE conference on Computational Intelligence and Games*. IEEE, 1–8.

[39] Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. 2017. Learning important features through propagating activation differences. In *International Conference on Machine Learning*. PMLR, 3145–3153.

[40] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. 2017. Smoothgrad: removing noise by adding noise. *arXiv preprint arXiv:1706.03825* (2017).

[41] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning*. JMLR, org, 3319–3328.

[42] Sarah Tan, Rich Caruana, Giles Hooker, Paul Koch, and Albert Gordo. 2018. Learning global additive explanations for neural nets using model distillation. *arXiv preprint arXiv:1801.08640* (2018).

[43] Jianrong Tao, Yu Xiong, Shiwei Zhao, Runze Wu, Xudong Shen, Tangjie Lyu, Changjie Fan, Zhipeng Hu, Sha Zhao, and Gang Pan. 2022. Explainable AI for cheating detection and churn prediction in online games. *IEEE Transactions on Games* (2022).

[44] Jianrong Tao, Yu Xiong, Shiwei Zhao, Yuhong Xu, Jianshi Lin, Runze Wu, and Changjie Fan. 2020. XAI-Driven Explainable Multi-view Game Cheating Detection. In *2020 IEEE Conference on Games (CoG)*. IEEE, 144–151.

[45] Markus Viljanen, Antti Airola, Jukka Heikkonen, and Tapio Pahikkala. 2017. Playtime measurement with survival analysis. *IEEE Transactions on Games* 10, 2 (2017), 128–138.

[46] Meng Xi, Zhiling Luo, Naibo Wang, Jianrong Tao, Ying Li, and Jianwei Yin. 2020. A Latent Feelings-aware RNN Model for User Churn Prediction with only Behaviour data. In *International Conference on Smart Data Services (SMDS)*. IEEE, 26–35.

[47] Carl Yang, Zhi Liu, Luo Jie, and Jiawei Han. 2018. I Know You'll Be Back: Interpretable New User Clustering and Churn Prediction on a Mobile Social Application. In *Proceedings of the 24th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*. ACM, 914–922.

[48] Liuyi Yao, Zhixuan Chu, Sheng Li, Yaliang Li, Jing Gao, and Aidong Zhang. 2020. A survey on causal inference. *arXiv preprint arXiv:2002.02770* (2020).

[49] Xiaobing Yu, Shunsheng Guo, Jun Guo, and Xiaorong Huang. 2011. An extended support vector machine forecasting framework for customer churn in e-commerce. *Expert Systems with Applications* 38, 3 (2011), 1425–1430.
