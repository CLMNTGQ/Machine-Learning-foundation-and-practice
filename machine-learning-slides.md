---
marp: true
theme: deftech3
size: 16:9
paginate: true
class: compact
header: 'Machine Learning — Foundations and Practice'
footer: '43.1240° N · 5.9280° E'
style: |
  section { --frame-right:calc(30% + 44px); --deftech-author:""; }
  section.full-width { display:flex; flex-direction:column; justify-content:center; align-items:stretch; padding:86px 44px 72px; text-align:left; font-size:24px; }
  section.full-width:is(.accent-band,.takeaway,.objectives) { justify-content:flex-start; padding-top:88px; }
  section.full-width > :is(h1,h2,h3):first-of-type { margin-top:0; }
  section.concept > h2:first-of-type { margin:0 0 16px; font-size:40px; }
  section.concept.image-right > p:first-of-type:has(> img:only-child) { inset:0 0 0 auto; height:100%; margin:0; }
  section.concept > header, section.full-width > header { top:36px; height:33px; left:44px; right:var(--frame-right); border:0; padding:0; font:400 16px/1.25 var(--deftech-sans); }
  section.concept > footer, section.full-width > footer { bottom:20px; height:32px; left:0; right:0; border:0; font-size:12px; text-align:center; }
  section.concept > footer::before, section.full-width > footer::before { left:128px; font-size:12px; }
  section > footer::after { display:none; content:none; }
  section.concept::after, section.full-width::after { top:auto; bottom:23px; left:44px; right:auto; width:68px; border:0; padding:0; text-align:right; font:400 12px/17px var(--deftech-mono); content:attr(data-marpit-pagination); }
  section.concept[data-marpit-pagination-total]::after, section.full-width[data-marpit-pagination-total]::after { content:attr(data-marpit-pagination) " / " attr(data-marpit-pagination-total); }
  section.concept.watermark { padding-inline:44px; isolation:isolate; justify-content:center; }
  section.concept.watermark > p:first-of-type:has(> img:only-child) { inset:80px 0 66px; width:100%; opacity:1; z-index:-1; }
  section.concept.watermark > :is(p,ul,ol,h1,h2,h3,blockquote), section.concept.watermark strong { color:#fff; }
---

<!-- _class: full-width -->
<!-- _paginate: false -->

# Machine Learning 
# Foundations and Practice

16 hours · CPU only · open-source tools

**Clément Gicquel** · AI Solution Architect

---

<!-- _class: full-width -->
## Course map

1. Roadmap & objectives [0]
2. Foundations and data preparation [1–2]
3. Linear models and honest evaluation [3–4]
4. Trees, ensembles and model selection [5–6]
5. Clustering, reduction and anomaly detection [7–8]
6. Neural networks and transferable representations [9–10]
7. Trustworthy ML and productization [11–12]
8. Group project [13]

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 00](images/generic/transitions-chapitre/introduction/transitions-centrees--00-introduction.jpg)

---

<!-- _class: full-width -->
<!-- header: '[0] — Course Roadmap & Learning Objectives' -->

# 0. Course Roadmap & Learning Objectives

### Format, objectives, assessment

---

<!-- _class: full-width -->

## 0. Format and breakdown

**2 days × 8 hours**, lectures and hands-on labs. The two days are one week apart.

| Day 1 | Hours | Day 2 | Hours |
|---|---|---|---|
| M1 Foundations & data [1–2] | 2h30 | M4 Unsupervised learning [7–8] | 2h30 |
| M2 Linear models & evaluation [3–4] | 3h | M5 Neural representations [9–10] | 2h30 |
| M3 Trees, ensembles & selection [5–6] | 2h30 | M6 Trustworthy ML & productization [11–12] | 2h |
| | | M7 Group project [13] | 1h |

*Week between the two days: consolidate the supervised-learning pipeline and frame the group project.*

---

<!-- _class: full-width objectives -->

## 0. Learning objectives

By the end of the course you can:

- Formulate a problem as a machine learning task
- Prepare data without leaking information
- Train the main model families
- Evaluate a model reliably
- Explain the maths behind the core algorithms
- Assess trust: explainability, fairness, robustness
- Package a model and serve it

---

<!-- _class: full-width -->

## 0. Prerequisites and assessment

**Prerequisites**
- Python: functions, lists, loops
- Linear algebra: vectors, matrices, matrix product
- Calculus: derivative, gradient
- Probability: mean, variance

> Notation is recalled when needed

**Assessment**
- **MCQ** [1–12]
- **Group project** [13]

---

<!-- _class: full-width -->

## 0. Datasets (open source)

| Dataset | Problem |
|---|---|
| **Titanic** | Classification, 891 passengers |
| **California Housing** | Regression, 20 640 districts |
| **FinBench `cf1`** | Rare frauds (0.67 %) |
| **Wine** | 3 classes; with and without labels |
| **Adult Census** | Fairness analysis |

---

<!-- _class: full-width -->

## 0. Setup

- **Google Colab**: nothing to install
- or a local **virtual environment**
- Libraries: pandas, scikit-learn, XGBoost, PyTorch (CPU), SHAP
- Data loaded through one file: `course_data.py`

---

<!-- _class: full-width -->

# Day 1
## Foundations and supervised learning

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 01](images/generic/transitions-chapitre/laboratoire-robotique/transitions-centrees--01-laboratoire-robotique--6e06a0b2d3ce.jpg)

---

<!-- _class: full-width -->
<!-- header: '[1] — What is Machine Learning?' -->

# 1. What is machine learning?

### Paradigms, vocabulary and the learning problem

---

<!-- _class: full-width objectives -->

## 1.1 Objectives

- **Define** machine learning
- **Name** the learning paradigms
- **Describe** the stages of an ML project
- **State** the formal setup: data, model, loss

---

<!-- _class: concept image-right -->


![Titanic](images/ml/exemples/titanic/recadrages-verticaux--titanic--54cce2768819.jpg)

## 1.2 Question: who survived the Titanic?

- Write **three rules** that predict survival
- Test them on the passengers
- Compare with a model in 15 minutes

> Hand-written rules vs rules learned from data

---


<!-- _class: concept watermark takeaway -->
<!-- header: 'Machine Learning — Foundations and Practice' -->

![Titanic](images/ml/exemples/titanic/slides--titanic--f4355a02f1b6.jpg)

## 1.2 Question: who survived the Titanic?

- Write **three rules** that predict survival
- Test them on the passengers
- Compare with a model in 15 minutes

> Hand-written rules vs rules learned from data

---

<!-- _class: concept image-right -->

![apprentissage supervise](images/ml/illustrations/apprentissage-supervise/portraits-spatiaux-essais--02-apprentissage-supervise--cf3b708d2136.jpg)

## 1.3 Classical programming vs machine learning

- Classical: **rules + data → answers**
- Machine learning: **data + answers → rules**
- The rules found are the **model**

![Diagram of supervised learning](images/ml/schemas/supervised-learning/deftech--01-supervised-learning--c13edc9aa7ce.png)

> ML replaces hand-written rules by rules estimated from data

---

<!-- _class: concept image-right -->

![apprentissage supervise](images/ml/illustrations/mitchell/mitchell2.jpg)

## 1.4 A precise definition (Mitchell, 1997)

A program **learns** from experience **E** on a task **T** with performance **P** if its P at T improves with E.

- **Spam filter**
- T: classify an email
- E: emails already labelled
- P: share correctly classified on **new** emails

---

<!-- _class: concept image-right -->

![contrat de prediction](images/ml/illustrations/contrat-de-prediction/portraits-spatiaux-essais--01-contrat-de-prediction--26b0f75ac076.jpg)

## 1.5 Essential vocabulary

| Term | Meaning |
|---|---|
| **Sample** | One row |
| **Feature** | An input variable |
| **Label** | What we want to predict |
| **Model** | Function: features → prediction |
| **Loss** | How wrong a prediction is |

---

<!-- _class: concept image-right -->

![optimization](images/ml/illustrations/optimization/deftech--08-optimization--d29c449ea175.jpg)

## 1.6 Parameters vs hyperparameters

- **Parameters**: learned from data (weights)
- **Hyperparameters**: chosen by you before training (tree depth, learning rate)
- **Training**: finding good parameters
- **Inference**: predicting on new inputs

> Parameters are learned. Hyperparameters are tuned.

---

<!-- _class: concept image-right -->

![apprentissage non supervise](images/ml/illustrations/unsupervised-learning/deftech-v2--02-unsupervised-learning--50d4744dda7b.jpg)

## 1.7 Learning paradigms

| Paradigm | Data | Example |
|---|---|---|
| **Supervised** | Inputs and labels | House price, spam |
| **Unsupervised** | Inputs only | Customer groups |
| **Self-supervised** | Labels made from the data | Masked word |
| **Reinforcement** | Rewards | A game-playing agent |

---

<!-- _class: concept image-right -->

![ml lifecycle](images/ml/illustrations/lifecycle/lifecycle1.png)

## 1.8 The life cycle of an ML project

**Define** → **Collect and explore** → **Prepare** → **Train** → **Evaluate** → **Deploy and monitor**

- **60–80 %** of the time goes into data
- The cycle is **iterative**: evaluation sends you back to the data

![Diagram of ml lifecycle](images/mlops/schemas/ml-lifecycle/deftech--01-ml-lifecycle--1ca23a176ee6.png)

---

<!-- _class: concept image-right -->
![risque empirique](images/ml/illustrations/variables-a-l-inference/portraits-navals--09-variables-a-l-inference--54bce308063f.jpg)

## 1.9 The formal setup

Dataset: $(x_i, y_i)$, model $f_\theta$, loss $\ell$

$$\hat\theta = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x_i), y_i\big)$$



> Learning means minimising the average loss on the training data.

---

<!-- _class: concept image-right -->
![train test split](images/ml/illustrations/train-test-split/deftech--03-train-test-split--53873342ff86.jpg)

## 1.10 The central difficulty

- We minimise the error on the data we **have**
- We care about the error on data we **will see**
- The gap between the two is the subject of **Section 4**

> Performance on new data is what matters : A model can exploit accidental patterns in a finite training set. It can also fail because future conditions no longer resemble the past.

---
<!-- _class: concept image-right invert -->
![modele de reference](images/ml/illustrations/modele-de-reference/portraits-spatiaux-essais--07-modele-de-reference--b45774a5cc11.jpg)

## 1.11 No miracle algorithm

- **No Free Lunch**: no algorithm is best on every problem
- Always start with a **simple baseline**
- Add complexity only if it **measurably** helps

---

<!-- _class: full-width accent-band -->

## 1. Exercises

📝 **1.1** Classify four use cases by learning paradigm and task type
📝 **1.2** Separate parameters from hyperparameters in a neural network
📝 **1.3** Write Mitchell's **T, E, P** for handwritten-digit recognition

---

<!-- _class: full-width takeaway -->

## [1] — Key Takeaways

- Machine learning estimates rules from examples by minimising a loss
- Performance on new data matters more than training performance
- Parameters are learned; hyperparameters are tuned
- Start with a simple baseline and add complexity only when it helps

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 02](images/generic/transitions-chapitre/conteneur-suspendu/transitions-centrees--02-conteneur-suspendu--9bc32c12124d.jpg)

---

<!-- _class: full-width -->
<!-- header: '[2] — Data and Preprocessing' -->

# 2. Data and preprocessing

### Exploration, transformations and leakage-safe pipelines

---

<!-- _class: full-width objectives -->

## 2.1 Objectives

- **Explore** a dataset and spot problems
- **Encode** and **scale** features
- **Split** data correctly
- **Avoid data leakage**
- Build a **pipeline**

---

<!-- _class: concept image-right -->

![variables a l inference](images/ml/illustrations/type/type1.png)

## 2.2 Types of data

| Kind | Example | Entry into a model |
|---|---|---|
| **Numerical** | Age | Directly |
| **Categorical** | City | Encode as numbers |
| **Ordinal** | Low / medium / high | Ordered integers |
| **Text, images** | Reviews, photos | Embeddings (Section 10) |
| **Time series** | Sensors | Split by time |

---

<!-- _class: concept image-right -->

![data scientist](images/generic/metiers/data-scientist/metiers--02-data-scientist--4d0acf9a6c2a.jpg)

## 2.3 Look at the data first

- **Shape**, types, first rows
- **Missing values** per column
- **Duplicates**
- **Distributions** (histograms)
- Impossible values? Balanced classes? Suspicious columns?

> No modelling before exploration

---

<!-- _class: concept image-right -->

![imputation](images/ml/illustrations/imputation/portraits-spatiaux-essais--10-imputation--cd5769d34e53.jpg)

## 2.4 Cleaning

- **Missing values**: drop, or **impute** (median is robust)
- Add a column "was missing": missingness can be informative
- **Outliers**: error or real? Keep real ones
- **Duplicates**: remove, or they land in train **and** test

---

<!-- _class: concept image-right -->

![one hot](images/ml/illustrations/one-hot/portraits-spatiaux--11-one-hot--73cbf9b02de4.jpg)

## 2.5 Encoding categorical variables

- **One-hot**: one binary column per category
- **Ordinal**: only if the order is meaningful

| city | Paris | Lyon | Nice |
|---|---|---|---|
| Paris | 1 | 0 | 0 |
| Lyon | 0 | 1 | 0 |

---

<!-- _class: concept image-right -->

![standardisation](images/ml/illustrations/standardisation/portraits-spatiaux-essais--12-standardisation--e63159ce0870.jpg)

## 2.6 Scaling numerical variables

- **Standardisation**: $z = (x - \mu)/\sigma$
- **Min–max**: maps values to $[0, 1]$
- **Needed** by: k-NN, SVM, k-means, gradient-based models
- **Not needed** by: trees

![Diagram of feature scaling](images/ml/schemas/feature-scaling/deftech--05-feature-scaling--42fa013eef19.png)

> A feature in the thousands would dominate a feature in 0–1

---

<!-- _class: concept image-right -->

![train test split](images/ml/illustrations/train-test-split/deftech--03-train-test-split--53873342ff86.jpg)

## 2.7 Splitting the data

| Set | Share | Role |
|---|---|---|
| **Training** | 60–70 % | Fit the parameters |
| **Validation** | 15–20 % | Compare models, tune |
| **Test** | 15–20 % | Final estimate, **used once** |

Choose the split for the prediction task

![train test split strategies](images/ml/schemas/train-test-split/deftech--03-train-test-split--9682193ba84c.png)

>Decide what must stay unseen at evaluation time.
---

<!-- _class: concept image-right -->

![fuite par pretraitement](images/ml/illustrations/fuite-par-pretraitement/portraits-spatiaux--18-fuite-par-pretraitement--4f96674fec99.jpg)

## 2.8 Data leakage

**Information that would not exist at prediction time slips into training.**

- Preprocessing **before** splitting
- A feature **derived from the target**
- **Duplicates** across sets
- **Future** information

![Diagram of data leakage](images/ml/schemas/data-leakage/deftech--04-data-leakage--ca2f4de2cd5c.png)

> Great test score, terrible real-world performance

---

<!-- _class: concept image-right -->

![fuite par pretraitement](images/ml/illustrations/fuite-par-pretraitement/portraits-spatiaux--18-fuite-par-pretraitement--4f96674fec99.jpg)

## 2.9 Question: find the bug

- A pipeline scales the data, **then** splits
- The test score is surprisingly good

Where is the leak? What does it change?

> Fit preprocessing on the training data only.

---

<!-- _class: concept image-right -->

![data engineer](images/generic/metiers/data-engineer/metiers--03-data-engineer--fdcd274cb95c.jpg)

## 2.10 Feature engineering

- **Ratios**: price per square metre
- **Dates**: day of week, month
- **Logarithms** for skewed variables
- **Interactions** and **aggregates**

> Often the best way to improve a classical model

---

<!-- _class: concept image-right -->

![pipeline de pretraitement](images/ml/illustrations/pipeline-de-pretraitement/portraits-travail--19-pipeline-de-pretraitement--5d6395d68d09.jpg)

## 2.11 The pipeline

```python
model = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression())])
model.fit(X_train, y_train)     # fitted on train only
model.score(X_test, y_test)
```

- Preprocessing is **re-fitted** inside cross-validation
- Compare with a **baseline** (`DummyClassifier`)

---

<!-- _class: full-width accent-band -->

## 2. Exercises

📝 **2.1** Explore the data
📝 **2.2** Handle missing ages
📝 **2.3** Build the pipeline
📝 **2.4** Compare with the baseline

---

<!-- _class: full-width takeaway -->

## [2] — Key Takeaways

- Explore and split the data before learning any transformation
- Fit imputation, encoding and scaling on training data only
- Keep preprocessing and the model in one leakage-safe pipeline
- Use validation for choices and the test set once

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 03](images/generic/transitions-chapitre/puce-electronique/transitions-centrees--03-puce-electronique--a052890f91c0.jpg)

---

<!-- _class: full-width -->
<!-- header: '[3] — Linear Models and Gradient Descent' -->

# 3. Linear models and gradient descent

### Regression, optimisation and regularisation

---

<!-- _class: full-width objectives -->

## 3.1 Objectives

- Fit and interpret **linear** and **logistic** regression
- Explain **gradient descent**
- Explain **regularisation** (L1, L2)

---

<!-- _class: concept image-right -->

![regression lineaire et residus](images/ml/illustrations/regression-lineaire-et-residus/portraits-travail--20-regression-lineaire-et-residus--8aca964b9cfb.jpg)

## 3.2 Linear regression

$$\hat y = w^\top x + b$$

- **Weights** $w$: one per feature
- **Bias** $b$: intercept
- $w_j$ = change in prediction for **one unit** of $x_j$, others being equal

![Diagram of linear regression residuals](images/ml/schemas/linear-regression-residuals/deftech--06-linear-regression-residuals--0b9914dd3a3a.png)

---

<!-- _class: concept image-right -->

![erreur quadratique moyenne](images/ml/illustrations/erreur-quadratique-moyenne/portraits-travail--21-erreur-quadratique-moyenne--903c9927b4e3.jpg)

## 3.3 The loss: mean squared error

$$L(w,b) = \frac{1}{n}\sum_{i=1}^{n}\big(w^\top x_i + b - y_i\big)^2$$

- Points $(1,2)$, $(2,3)$, $(3,5)$
- Best line: $\hat y = 1.5x + 0.33$
- MSE $\approx 0.056$



---

<!-- _class: concept image-right -->

![linear regression residuals](images/ml/illustrations/linear-regression-residuals/deftech--06-linear-regression-residuals--9330fc7876f6.jpg)

## 3.4 The normal equation

$$\hat w = (X^\top X)^{-1} X^\top y$$

- Set the gradient of the loss to **zero**
- The loss is a **convex bowl**: one minimum
- Needs $X^\top X$ **invertible**

> Proof in Annex A

---

<!-- _class: concept image-right -->

![descente de gradient](images/ml/illustrations/descente-de-gradient/portraits-spatiaux--24-descente-de-gradient--e7644df4335e.jpg)

## 3.5 Gradient descent

$$\theta \leftarrow \theta - \eta\,\nabla L(\theta)$$

- The **gradient** points uphill
- Move the **opposite** way
- $\eta$ = **learning rate**, the step size

![Diagram of gradient descent](images/ml/schemas/gradient-descent/deftech--09-gradient-descent--6174817a2ee4.png)

---

<!-- _class: concept image-right -->

![descente de gradient](images/ml/illustrations/gradient-descent/deftech-v2--09-gradient-descent--a629cc45cafa.jpg)

## 3.6 Gradient descent: a worked example

Minimise $L(w) = (w-3)^2$, start at $w_0 = 0$, $\eta = 0.1$

- Step 1: gradient $-6$ → $w = 0.6$
- Step 2: gradient $-4.8$ → $w = 1.08$
- … converges to **3**

---

<!-- _class: concept image-right -->

![taux d apprentissage](images/ml/illustrations/gradient-descent/deftech--09-gradient-descent--f2ef1ea65dd8.jpg)

## 3.7 The learning rate

- **Too small**: very slow
- **Too large**: oscillation or divergence
- The most important hyperparameter of gradient-based training

![learning rate](images/ml/schemas/learning-rate/serie-2--03-learning-rate--0a1248b7be6a.png)

---

<!-- _class: concept image-right -->

![mini lots et epoques](images/ml/illustrations/mini-lots-et-epoques/portraits-travail--51-mini-lots-et-epoques--652abea33241.jpg)

## 3.8 Batch, stochastic, mini-batch

| Variant | Samples per update |
|---|---|
| **Batch** | All $n$ |
| **Stochastic (SGD)** | 1 |
| **Mini-batch** | 32–256 |

> Mini-batch is the standard compromise

---

<!-- _class: concept image-right -->

![regression logistique et sigmoide](images/ml/illustrations/regression-logistique-et-sigmoide/portraits-spatiaux--22-regression-logistique-et-sigmoide--306801866ed4.jpg)

## 3.9 Logistic regression

$$\hat p = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

- A **classifier**, despite its name
- Output = a **probability**
- Predict class 1 if $\hat p \ge 0.5$ (the **threshold** can change)

![Diagram of logistic regression threshold](images/ml/schemas/logistic-regression-threshold/deftech--07-logistic-regression-threshold--bfbd3d7336d2.png)

---

<!-- _class: concept image-right -->

![log loss](images/ml/illustrations/logistic-regression-threshold/recadrages-verticaux--07-logistic-regression-threshold--3100e315f30c.jpg)

## 3.10 The log loss

$$\ell(\hat p, y) = -\big[y\log\hat p + (1-y)\log(1-\hat p)\big]$$

- Small when the true class gets a high probability
- **Huge** when the model is confidently wrong
- **Convex**: gradient descent finds the global minimum

---

<!-- _class: concept image-right -->

![regularisation l2](images/ml/illustrations/regularisation-l2/portraits-spatiaux-essais--26-regularisation-l2--c20652ce199f.jpg)

## 3.11 Regularisation

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | $\lambda\sum w_j^2$ | $\lambda\sum \lvert w_j\rvert$ |
| Effect | Shrinks weights | Sets many to **exactly 0** |
| Use | Correlated features | **Feature selection** |

![Diagram of ridge regularization](images/ml/schemas/ridge-regularization/deftech--12-ridge-regularization--348573ae94d9.png)

> Always standardise first

---

<!-- _class: full-width accent-band -->

## 3. Exercises

📝 **3.1** **Titanic**: logistic regression, read the weights
📝 **3.2** **California Housing**: OLS, Ridge, Lasso
📝 **3.3** **Autonomy**: code gradient descent in 10 lines

---

<!-- _class: full-width takeaway -->

## [3] — Key Takeaways

- Linear models are fast, interpretable baselines
- Gradient descent lowers loss by moving against its gradient
- Logistic regression outputs class probabilities
- L1 and L2 regularisation trade training fit for better generalisation

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 04](images/generic/transitions-chapitre/drone-sous-marin/transitions-centrees--04-drone-sous-marin--d1567aee0402.jpg)

---

<!-- _class: full-width -->
<!-- header: '[4] — Evaluating a Model' -->

# 4. Evaluating a model

### Generalisation, validation and decision metrics

---

<!-- _class: full-width objectives -->

## 4.1 Objectives

- Explain **overfitting**, underfitting and the **bias–variance** trade-off
- Use **cross-validation**
- Choose the right **metric**, even with imbalanced classes

---

<!-- _class: concept image-right -->

![selection et test final](images/ml/illustrations/selection-et-test-final/portraits-travail--36-selection-et-test-final--cdb223366024.jpg)

## 4.2 Generalisation

- **Training error**: on the data used for fitting
- **Generalisation error**: on new data
- Estimated on a **held-out** set



> A model is useful only if it works on new data.

---

<!-- _class: concept image-right -->

![sous et surajustement](images/ml/illustrations/sous-et-surajustement/portraits-travail--37-sous-et-surajustement--199df06da561.jpg)

## 4.3 Underfitting, good fit, overfitting

| Situation | Training | Validation |
|---|---|---|
| **Underfitting** | High | High |
| **Good fit** | Low | Low |
| **Overfitting** | Very low | **Much higher** |

![Diagram of overfitting](images/ml/schemas/overfitting/deftech--11-overfitting--a692f97fe42a.png)

---

<!-- _class: concept image-right -->

![overfitting](images/ml/illustrations/overfitting/deftech--11-overfitting--edb7519a3998.jpg)

## 4.4 Question: which polynomial degree?

- 60 noisy points of a sine curve
- Degree 1, 4 or 14?

Predict the **validation error** curve, then run it.

> Training error always falls. Validation error falls, then rises.

---

<!-- _class: concept image-right -->

![sous et surajustement](images/ml/illustrations/sous-et-surajustement/portraits-travail--37-sous-et-surajustement--199df06da561.jpg)

## 4.5 Bias and variance

$$\text{Error} = \text{bias}^2 + \text{variance} + \text{noise}$$

- **Bias**: model too simple
- **Variance**: model too sensitive to the training set
- **Noise**: cannot be removed

> Proof in Annex C

---

<!-- _class: concept image-right -->

![ridge regularization](images/ml/illustrations/ridge-regularization/deftech--12-ridge-regularization--733e8d2586ad.jpg)

## 4.6 Remedies

| High bias (underfitting) | High variance (overfitting) |
|---|---|
| More flexible model | More data |
| Better features | Simpler model |
| Less regularisation | More regularisation, bagging, early stopping |

---

<!-- _class: concept image-right -->

![cross validation](images/ml/illustrations/cross-validation/deftech--14-cross-validation--cb2b6a07d1b3.jpg)

## 4.7 Validation strategies

- **Hold-out**: one split, depends on luck
- **k-fold CV**: train $k$ times, average the scores ($k = 5$ or $10$)
- **Stratified**: keeps class proportions
- **Group / time-series** split when rows are not independent

![Diagram of cross validation](images/ml/schemas/cross-validation/deftech--14-cross-validation--b99c35e63905.png)

---

<!-- _class: concept image-right -->

![selection et test final](images/ml/illustrations/anomaly-detection/deftech--23-anomaly-detection--48095cdef8ec.jpg)

## 4.8 The golden rule

**The test set is used once, at the very end.**

- Tune on validation or cross-validation
- If you tune on the test set, its score is **no longer honest**

---

<!-- _class: concept image-right -->

![linear regression residuals](images/ml/illustrations/regression-lineaire-et-residus/portraits-travail--20-regression-lineaire-et-residus--8aca964b9cfb.jpg)

## 4.9 Regression metrics

| Metric | Remark |
|---|---|
| **MAE** | Robust to outliers |
| **MSE** | Penalises large errors |
| **RMSE** | Same unit as $y$ |
| **R²** | 1 = perfect, 0 = predicting the mean |



---

<!-- _class: concept image-right -->

![matrice de confusion](images/ml/illustrations/matrice-de-confusion/portraits-spatiaux--27-matrice-de-confusion--9ce775e4c1e4.jpg)

## 4.10 The confusion matrix

| | Predicted +  | Predicted − |
|---|---|---|
| **Actually +** | TP | FN |
| **Actually −** | FP | TN |

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** balances both

![class imbalance and metrics](images/ml/schemas/class-imbalance-and-metrics/deftech--13-class-imbalance-and-metrics--d625e07cacb6.png)

---

<!-- _class: concept image-right -->

![Finance](images/ml/exemples/fraude-bancaire/recadrages-verticaux--finance--07d749de92fd.jpg)

## 4.11 Why accuracy can lie

1 000 patients, 60 ill. TP 40, FP 10, FN 20, TN 930.

- Accuracy **0.97**
- Precision 0.80, recall 0.67, F1 0.73
- "Always healthy": accuracy **0.94**, recall **0**

![Diagram of class imbalance and metrics](images/ml/schemas/class-imbalance-and-metrics/deftech--13-class-imbalance-and-metrics--d625e07cacb6.png)

> Accuracy hides the problem

---

<!-- _class: concept image-right -->

![Finance](images/ml/exemples/fraude-bancaire/slides--finance--8e977c0ceb06.jpg)

## 4.12 Vote: is 99 % accuracy good?

- Fraud detection: **0.67 %** of transactions are fraud
- A model never flags fraud
- Its accuracy?

> About **99 %**, and it is useless

---

<!-- _class: concept image-right -->

![roc](images/ml/illustrations/roc/portraits-spatiaux--30-roc--be61e59470b2.jpg)

## 4.13 ROC and precision–recall

- Lowering the **threshold** raises recall and lowers precision
- **ROC / AUC**: 0.5 = random, 1 = perfect
- **Precision–recall** curve: better when positives are rare

![roc et precision rappel](images/ml/schemas/roc-et-precision-rappel/schema-revu--roc-et-precision-rappel.png)

---

<!-- _class: concept image-right -->

![cout des erreurs](images/ml/illustrations/cout-des-erreurs/portraits-spatiaux-essais--08-cout-des-erreurs--0f8d50e16575.jpg)

## 4.14 Choose by the cost of errors

- Missing a cancer (FN) is not a false alarm (FP)
- Smoke detector: favour **recall**
- Spam folder: favour **precision**



> Choose the metric and the threshold from the cost, not by habit

---

<!-- _class: concept image-right -->

![desequilibre des classes](images/ml/illustrations/class-imbalance-and-metrics/recadrages-verticaux--13-class-imbalance-and-metrics--c130e5c63da6.jpg)

## 4.15 Imbalanced classes

- Use **precision, recall, F1, PR-AUC**
- **Class weights**: penalise errors on the rare class
- **Resample** the training folds only
- Tune the **threshold**

![Diagram of class imbalance and metrics](images/ml/schemas/class-imbalance-and-metrics/deftech--13-class-imbalance-and-metrics--d625e07cacb6.png)

---

<!-- _class: concept image-right -->

![desequilibre des classes](images/ml/illustrations/cross-validation/recadrages-verticaux--14-cross-validation--b7c49fc3f5cd.jpg)

## 4.16 Cross validation

- **Cross validation** is a technique for estimating the **generalisation performance** of a model
- Split your data into folds, train and validate on different splits, and report the mean and spread of performance metrics.
- Use **stratified folds for imbalanced classes** to ensure each fold has a representative distribution.

---

<!-- _class: full-width accent-band -->

## 4. Exercises

📝 **4.1** **Titanic**: cross-validation, mean **and spread**
📝 **4.2** **FinBench `cf1`**: baseline, class weights, recall of frauds

---

<!-- _class: full-width takeaway -->


## [4] — Key Takeaways

- Generalisation performance, not training error, determines usefulness
- Select models with validation and report cross-validation mean and spread
- Use the test set once, after all choices are complete
- Choose metrics and thresholds from the cost of errors
- Accuracy can hide failure on imbalanced data

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 05](images/generic/transitions-chapitre/decollage/transitions-centrees--05-decollage--f2674f72d02b.jpg)

---

<!-- _class: full-width -->
<!-- header: '[5] — Decision Trees and Ensembles' -->

# 5. Decision trees and ensembles

### Splits, bagging and boosting

---

<!-- _class: full-width objectives -->

## 5.1 Objectives

- Explain how a tree chooses its **splits**
- Explain how **bagging** and **boosting** fix a single tree
- Choose between random forest and gradient boosting

---

<!-- _class: concept image-right -->

![arbre de decision](images/ml/illustrations/arbre-de-decision/portraits-travail--38-arbre-de-decision--43212a76937c.jpg)

## 5.2 A decision tree

- A sequence of **yes/no questions** on the features
- **Leaves** hold the prediction
- Built **greedily**: at each node, take the best split

![Diagram of decision trees](images/ml/schemas/decision-trees/deftech--16-decision-trees--883a64b0143d.png)

> Easy to read, no scaling needed

---

<!-- _class: concept image-right -->

![impurete de gini](images/ml/illustrations/impurete-de-gini/portraits-travail--39-impurete-de-gini--68829597eecd.jpg)

## 5.3 Impurity

- **Gini**: $G = 1 - \sum_k p_k^2$
- **Entropy**: $H = -\sum_k p_k\log_2 p_k$
- 0 for a **pure** node, maximal when mixed
- Best split = largest **impurity decrease**

---

<!-- _class: concept image-right -->

![impurete de gini](images/ml/illustrations/impurete-de-gini/portraits-travail--39-impurete-de-gini--68829597eecd.jpg)

## 5.4 Worked example: a Gini split

- Node: 6 positive, 4 negative → $G = 0.48$
- Left: 5 positive, 0 negative → $G = 0$
- Right: 1 positive, 4 negative → $G = 0.32$
- After the split: $0.5\times0 + 0.5\times0.32 = 0.16$
- **Gain = 0.32**

![gini split](images/ml/schemas/gini-split/schema-revu--gini-split.png)

---

<!-- _class: concept image-right -->

![controle de la profondeur](images/ml/illustrations/controle-de-la-profondeur/portraits-travail--40-controle-de-la-profondeur--fbd227d016af.jpg)

## 5.5 A single tree overfits

- A deep tree **memorises** the training set
- A small change in the data gives a very different tree
- Controls: `max_depth`, `min_samples_leaf`, **pruning**



> High variance

---

<!-- _class: concept image-right -->

![bagging vs boosting](images/ml/illustrations/bagging-vs-boosting/deftech--17-bagging-vs-boosting--38efcb6ed50e.jpg)

## 5.6 Bagging

- Draw $B$ **bootstrap samples** (with replacement)
- Train one deep tree on each
- **Average** (regression) or **vote** (classification)

> Averaging many noisy, different predictors reduces variance.

---

<!-- _class: concept image-right -->

![random forest](images/ml/illustrations/random-forest/portraits-travail--41-random-forest--40028ddb3d9c.jpg)

## 5.7 Random forest

- Bagging + **random subset of features** at each split
- Makes trees **less correlated**
- Variance of the average: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$

> Adding trees never removes the first term. Decorrelate.

---

<!-- _class: concept image-right -->

![gradient boosting](images/ml/illustrations/gradient-boosting/portraits-travail--42-gradient-boosting--030c75cf4158.jpg)

## 5.8 Boosting

- Build trees **one after another**
- Each small tree corrects the **errors** so far
- Add it with a small **learning rate**
- Reduces mainly **bias**

> Needs early stopping

---

<!-- _class: concept image-right -->

![bagging vs boosting](images/ml/illustrations/bagging-vs-boosting/deftech--17-bagging-vs-boosting--38efcb6ed50e.jpg)

## 5.9 Random forest vs gradient boosting

| | Random forest | Gradient boosting |
|---|---|---|
| **Trees** | Deep, independent | Shallow, sequential |
| **Reduces** | Variance | Bias |
| **Tuning** | Low | Higher |
| **Overfitting** | Low risk | Use early stopping |

![Diagram of bagging vs boosting](images/ml/schemas/bagging-vs-boosting/deftech--17-bagging-vs-boosting--d5e23bfc575a.png)

---

<!-- _class: concept image-right -->

![importance par permutation](images/ml/illustrations/importance-par-permutation/portraits-travail--43-importance-par-permutation--cd405ca1bf6d.jpg)

## 5.10 Feature importance

- Default **impurity-based** importance is biased
- Prefer **permutation importance** on held-out data
- More in Section 11

![Diagram of explainability](images/mlops/schemas/explainability/deftech--29-explainability--4d7a4c3e9594.png)

---

<!-- _class: full-width accent-band -->

## 5. Exercises

📝 **5.1** **Titanic**: tree depth 3, unlimited tree, forest, boosting
📝 **5.2** **California Housing**: Ridge, forest, boosting (RMSE)
📝 **5.3** **Autonomy**: plot score against the number of trees

---

<!-- _class: full-width takeaway -->

## [5] — Key Takeaways

- A single tree is readable and flexible but unstable
- Bagging diverse trees reduces variance through averaging
- Random forests decorrelate trees with random feature subsets
- Boosting corrects errors sequentially and mainly reduces bias
- Compare ensembles with a simple baseline

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 06](images/generic/transitions-chapitre/ecran-pixels/transitions-centrees--06-ecran-pixels--3fda77f3bd17.jpg)

---

<!-- _class: full-width -->
<!-- header: '[6] — k-NN, SVM and Model Selection' -->

# 6. k-NN, SVM and model selection

### Distance, margins and leakage-safe comparison

---

<!-- _class: full-width objectives -->

## 6.1 Objectives

- Explain **k-NN** and **SVM**, and their limits
- **Tune** hyperparameters without leaking data
- Pick a reasonable **first model**

---

<!-- _class: concept image-right -->

![k nearest neighbours](images/ml/illustrations/k-nearest-neighbours/deftech-v2--18-k-nearest-neighbours--c71d9c17db32.jpg)

## 6.2 k-nearest neighbours

- Find the **$k$ most similar** training samples
- They **vote** (or average)
- **No training phase**: the model is the dataset

![Diagram of k nearest neighbours](images/ml/schemas/k-nearest-neighbours/deftech--18-k-nearest-neighbours--3055b27b1924.png)

> Small $k$ overfits. Large $k$ is smoother.

---

<!-- _class: concept image-right -->

![feature scaling](images/ml/illustrations/feature-scaling/deftech--05-feature-scaling--15b96bc12a86.jpg)

## 6.3 Scaling is essential for k-NN

- Wine: 13 measurements with ranges differing by about **1 000**
- Raw features: accuracy about **0.7**
- Scaled features: accuracy about **0.95**

![standardization and distance](images/ml/schemas/standardization-and-distance/serie-2--01-standardization-and-distance--b9a9851568e3.png)

> Distances are dominated by large-scale features

---

<!-- _class: concept image-right -->

![k nearest neighbours](images/ml/illustrations/k-nearest-neighbours/deftech--18-k-nearest-neighbours--23c63beef6e3.jpg)

## 6.4 The curse of dimensionality

- In high dimension, all points are **almost equally far**
- "Nearest" loses its meaning
- k-NN works best with **few features**

---

<!-- _class: concept image-right -->

![support vector machines](images/ml/illustrations/support-vector-machines/deftech--19-support-vector-machines--92461d0e71a1.jpg)

## 6.5 Support vector machines

- Choose the hyperplane with the **largest margin**
- Only the **support vectors** matter
- Margin width $= 2/\lVert w\rVert$

![Diagram of support vector machines](images/ml/schemas/support-vector-machines/deftech--19-support-vector-machines--182789ea22b8.png)

> Proof in Annex G

---

<!-- _class: concept image-right -->

![support vector machines](images/ml/illustrations/support-vector-machines/deftech--19-support-vector-machines--92461d0e71a1.jpg)

## 6.6 Soft margin and the parameter C

- Real data is rarely separable
- **Large C**: few violations tolerated, risk of overfitting
- **Small C**: wider, more tolerant margin

---

<!-- _class: concept image-right -->

![kernel intuition](images/ml/illustrations/kernel-intuition/deftech--20-kernel-intuition--adbf58e881cd.jpg)

## 6.7 The kernel trick

- Separate classes that are **not linearly separable**
- Replace dot products by a **kernel**
- **RBF**: $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$
- Large $\gamma$: wiggly boundary, **overfitting**

![svm nonlinear feature map](images/ml/schemas/svm-nonlinear-feature-map/serie-2--06-svm-nonlinear-feature-map--572b8566b847.png)

---

<!-- _class: concept image-right -->

![selection et test final](images/ml/illustrations/selection-et-test-final/portraits-travail--36-selection-et-test-final--cdb223366024.jpg)

## 6.8 Hyperparameter tuning

| Strategy | When |
|---|---|
| **Grid search** | Few hyperparameters |
| **Random search** | Many hyperparameters |
| **Bayesian optimisation** | Expensive models |

![Diagram of cross validation](images/ml/schemas/cross-validation/deftech--14-cross-validation--b99c35e63905.png)

> Cross-validate on the training data; the test set stays untouched.

---

<!-- _class: concept image-right -->

![data scientist](images/generic/metiers/data-scientist/metiers--02-data-scientist--4d0acf9a6c2a.jpg)

## 6.9 Which model first?

| Situation | First choice |
|---|---|
| Interpretability, small data | Regularised linear, shallow tree |
| Tabular, best accuracy | Gradient boosting, random forest |
| Small data, high dimension | SVM (RBF), regularised linear |
| Images, text, audio | Neural networks, pre-trained |

---

<!-- _class: full-width accent-band -->

## 6. Exercises

📝 **6.1** In pairs, use **Wine** and **Titanic**
📝 **6.2** Compare logistic regression, k-NN, SVM, forest and boosting
📝 **6.3** Compare **with and without** scaling
📝 **6.4** Post your **scores** on the shared board

---

<!-- _class: full-width takeaway -->

## [6] — Key Takeaways

- No model family is universally best
- Scale features for k-NN and SVM
- Compare a few families under the same cross-validation pipeline
- Tune on training data and touch the test set only once

---

<!-- _class: full-width -->

# One week
## Homework and project kick-off

---

<!-- _class: concept image-right -->

![machine learning engineer](images/generic/metiers/machine-learning-engineer/metiers--04-machine-learning-engineer--28b23eecc003.jpg)

## Homework

- **Revise** Sections 2 to 4
- **Mini-lab**: a dataset you have not used (another FinBench subset, or any tabular dataset of the Hub)
- Workflow: **EDA → pipeline → three model families → CV comparison → short conclusion**
- **Start the group project**
- **Re-read** derivative and matrix product

---

<!-- _class: full-width -->

# Day 2
## Unsupervised learning, neural networks and responsible ML

---

<!-- _class: concept image-right -->

![data scientist](images/generic/metiers/data-scientist/metiers--02-data-scientist--4d0acf9a6c2a.jpg)

## Recap quiz: Day 1

1. Why **split before** preprocessing?
2. What are the symptoms of **overfitting**?
3. When does **accuracy lie**?
4. What does **regularisation** do?
5. **Bagging** or **boosting**: which cuts variance?

> Answer alone for 2 minutes, then compare with your neighbour

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 07](images/generic/transitions-chapitre/assemblage-robot/transitions-centrees--07-assemblage-robot--3f7a852e7d1e.jpg)

---

<!-- _class: full-width -->
<!-- header: '[7] — Clustering' -->

# 7. Clustering

### Structure, geometry and validation without labels

---

<!-- _class: full-width objectives -->

## 7.1 Objectives

- Explain **unsupervised learning** and why it is hard to evaluate
- Run **k-means**, **DBSCAN**, **Gaussian mixtures**
- **Choose k** and evaluate without labels

---

<!-- _class: concept image-right -->

![apprentissage non supervise](images/ml/illustrations/apprentissage-non-supervise/portraits-travail--44-apprentissage-non-supervise--1b683b68e25d.jpg)

## 7.2 The clustering problem

- Samples with **no labels**
- Group them: **similar inside**, **different between** clusters
- The result depends on the **similarity** and the **algorithm**

![Diagram of unsupervised learning](images/ml/schemas/unsupervised-learning/deftech--02-unsupervised-learning--faf29f1396da.png)

> Always scale the features first

---

<!-- _class: concept image-right -->

![k means](images/ml/illustrations/k-means/portraits-travail--45-k-means--380736ec2843.jpg)

## 7.3 k-means

1. **Initialise** $k$ centroids
2. **Assign** each sample to its nearest centroid
3. **Update** each centroid to the mean of its samples
4. **Repeat** until nothing changes

![kmeans iteration](images/ml/schemas/kmeans-iteration/premieres-figures--07-01-kmeans-iteration--1c7448199388.png)

> Minimises the **inertia**: sum of squared distances to the centroid

---

<!-- _class: concept image-right -->

![k means](images/ml/illustrations/k-means/portraits-travail--45-k-means--380736ec2843.jpg)

## 7.4 k-means: a worked example

- Samples $\{1, 2, 3, 10, 11, 12\}$, $k = 2$
- Centroids: **2** and **11**
- Inertia $= 2 + 2 = 4$

> Practice at the board: 6 points, 2 iterations

---

<!-- _class: concept image-right -->

![initialisation et stabilite](images/ml/illustrations/initialisation-et-stabilite/portraits-travail--46-initialisation-et-stabilite--7257b6b1ac16.jpg)

## 7.5 k-means: guarantees and limits

- Never increases the inertia, always **stops**
- But at a **local** minimum: run several times (`n_init`)
- You must choose **$k$**
- Assumes **spherical** clusters of similar size
- Sensitive to **outliers** and **scaling**

> Proof in Annex F

---

<!-- _class: concept image-right -->

![k means vs dbscan](images/ml/illustrations/k-means-vs-dbscan/deftech--21-k-means-vs-dbscan--0684b1177092.jpg)

## 7.6 Choosing $k$

- **Elbow**: inertia always falls; look for the bend
- **Silhouette**: $s = (b - a)/\max(a, b)$, from $-1$ to $1$
- **Domain validation**: do the clusters make sense? Are they stable?

![silhouette score](images/ml/schemas/silhouette-score/serie-2--07-silhouette-score--dca74dce231b.png)

---

<!-- _class: concept image-right -->

![Wine](images/ml/exemples/vignoble/slides--wine--2a891c02a013.jpg)

## 7.7 Question: how many wine cultivars?

- Wine: 178 wines, 13 measurements, **labels hidden**
- Predict $k$, then compute the silhouette for $k = 2$ to $6$

> Then reveal the real cultivars

---

<!-- _class: concept image-right -->

![Wine](images/ml/exemples/vignoble/slides--wine--2a891c02a013.jpg)

## 7.8 Comparing with known labels

- **Adjusted Rand index (ARI)**: do two partitions agree on pairs of samples?
- **1** = identical, about **0** = random
- Labels **judge** the clusters, they never **build** them
- Wine: k-means finds clusters close to the cultivars (ARI about **0.9**)

---

<!-- _class: concept image-right -->

![k means vs dbscan](images/ml/illustrations/k-means-vs-dbscan/deftech--21-k-means-vs-dbscan--0684b1177092.jpg)

## 7.9 DBSCAN

- Clusters = **dense regions** separated by sparse ones
- Two hyperparameters: `eps` and `min_samples`
- **Core**, **border** and **noise** points
- Finds **arbitrary shapes**, flags **outliers**
- Weak when **densities vary**

![Diagram of k means vs dbscan](images/ml/schemas/k-means-vs-dbscan/deftech--21-k-means-vs-dbscan--96e6c52fa8f7.png)

---

<!-- _class: concept image-right -->

![unsupervised learning](images/ml/illustrations/unsupervised-learning/deftech--02-unsupervised-learning--e9a45322ecaf.jpg)

## 7.10 Gaussian mixture models

- Data = a mix of **Gaussians**
- Elliptical clusters of different sizes
- **Soft** assignments: a probability per cluster
- Fitted by **EM**, number of components by **BIC**

---

<!-- _class: concept image-right -->

![k means vs dbscan](images/ml/illustrations/k-means-vs-dbscan/deftech--21-k-means-vs-dbscan--0684b1177092.jpg)

## 7.11 Which clustering algorithm?

| | k-means | DBSCAN | GMM |
|---|---|---|---|
| **Choose $k$** | Yes | No | Yes |
| **Shape** | Spherical | Arbitrary | Elliptical |
| **Outliers** | Sensitive | Detected | Sensitive |
| **Output** | Hard labels | Labels + noise | Probabilities |

![k means vs dbscan](images/ml/schemas/k-means-vs-dbscan/deftech--21-k-means-vs-dbscan--96e6c52fa8f7.png)

---

<!-- _class: full-width accent-band -->

## 7. Exercises

📝 **7.1** **Wine**: k-means with and without scaling, compare the ARI
📝 **7.2** **Half-moons**: observe k-means fail, then tune DBSCAN's `eps`

---

<!-- _class: full-width takeaway -->

## [7] — Key Takeaways

- Clustering proposes structure rather than discovering a unique truth
- Scale features before applying distance-based methods
- Match k-means, DBSCAN or GMM to the expected cluster geometry
- Validate with stability, internal metrics and domain usefulness

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 08](images/generic/transitions-chapitre/soufflerie/transitions-centrees--08-soufflerie--cb2fac332158.jpg)

---

<!-- _class: full-width -->
<!-- header: '[8] — Dimensionality Reduction and Anomaly Detection' -->

# 8. Dimensionality reduction and anomaly detection

### Compression, visualisation and unusual observations

---

<!-- _class: full-width objectives -->

## 8.1 Objectives

- Explain the **curse of dimensionality**
- Apply **PCA** and read the explained variance
- Use **t-SNE / UMAP** knowing their pitfalls
- Apply basic **anomaly detection**

---

<!-- _class: concept image-right -->

![pca](images/ml/illustrations/pca/deftech--22-pca--5f59256ee7af.jpg)

## 8.2 Why reduce dimension?

- Easier to **overfit** and to compute
- Distances **lose meaning**
- Humans see **2–3 dimensions** only
- Uses: visualisation, noise removal, compression, speed

---

<!-- _class: concept image-right -->

![pca](images/ml/illustrations/pca/portraits-navals--47-pca--ac4535889584.jpg)

## 8.3 PCA: the idea

- Find the axes along which the data **varies most**
- Keep the **first few**
- **Centre** (and scale), compute the **covariance matrix**, take its **eigenvectors**

![Diagram of pca](images/ml/schemas/pca/deftech--22-pca--88058ba5481b.png)

> First component = direction of maximal variance

---

<!-- _class: concept image-right -->

![pca](images/ml/illustrations/pca/deftech--22-pca--5f59256ee7af.jpg)

## 8.4 PCA: a worked example

$$\Sigma = \begin{pmatrix}3 & 1\\ 1 & 3\end{pmatrix}$$

- Eigenvalues: **4** and **2**
- First component $(1,1)/\sqrt2$ explains $4/6 \approx 67\,\%$

> Proof in Annex E

---

<!-- _class: concept image-right -->

![pca](images/ml/illustrations/pca/deftech--22-pca--5f59256ee7af.jpg)

## 8.5 How many components?

- Keep **90–95 %** of the cumulative explained variance
- or look at the **elbow** of the scree plot
- Wine: **2** components explain about **55 %**, **8** reach 90 %

---

<!-- _class: concept image-right -->

![pca](images/ml/illustrations/pca/deftech--22-pca--5f59256ee7af.jpg)

## 8.6 PCA: limits

- **Linear** only
- Components are **mixtures** of features: harder to read
- Sensitive to **scaling**
- Variance is not always what **matters** for a task

---

<!-- _class: concept image-right -->

![Wine](images/ml/exemples/vignoble/slides--wine--2a891c02a013.jpg)

## 8.7 Question: what will the PCA plot show?

- Wine, 13 features projected on **2 components**
- Colour the points by **cultivar**



> PCA never saw the labels, yet the cultivars separate

---

<!-- _class: concept image-right -->

![embeddings](images/ml/illustrations/embeddings/deftech--27-embeddings--ace4503931fe.jpg)

## 8.8 t-SNE and UMAP

- Keep **neighbours** close in 2D or 3D
- Excellent to **look** at embeddings and clusters
- Distances between clusters and cluster sizes are **not meaningful**
- Depends on **hyperparameters** and the **seed**

> Use t-SNE and UMAP to explore, not to prove.

---

<!-- _class: concept image-right -->

![anomaly detection](images/ml/illustrations/anomaly-detection/deftech-v2--23-anomaly-detection--20c4c9e6cc7c.jpg)

## 8.9 Anomaly detection

| Method | Principle |
|---|---|
| **Statistical** | Far from the mean, $\lvert z\rvert > 3$ |
| **Isolation Forest** | Anomalies are isolated in **few splits** |
| **LOF** | Lower density than neighbours |
| **One-class SVM** | Boundary around "normal" |
| **Autoencoder** | Large reconstruction error |

![Diagram of anomaly detection](images/ml/schemas/anomaly-detection/deftech--23-anomaly-detection--1c85bf78cb3d.png)

---

<!-- _class: concept image-right -->

![Finance](images/ml/exemples/fraude-bancaire/slides--finance--8e977c0ceb06.jpg)

## 8.10 Isolation Forest on card fraud

- **FinBench `cf1`**, no labels given to the model
- Do **frauds** get the highest scores?
- Metric: **average precision**; random = share of frauds (about 0.007)



> Evaluation is hard: involve domain experts

---

<!-- _class: full-width accent-band -->

## 8. Exercises

📝 **8.1** **Wine**: PCA, cumulative variance, 2D plot by cultivar
📝 **8.2** **FinBench `cf1`**: Isolation Forest vs LOF, top 100 scores

---

<!-- _class: full-width takeaway -->

## [8] — Key Takeaways

- PCA compresses linearly while preserving as much variance as possible
- Scaling matters for PCA and distance-based methods
- t-SNE and UMAP preserve local neighbourhoods, not global distances
- Anomaly methods rank unusual points and require labels or domain review

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 09](images/generic/transitions-chapitre/antenne-polaire/transitions-centrees--09-antenne-polaire--5c3ed38bb697.jpg)

---

<!-- _class: full-width -->
<!-- header: '[9] — Neural Networks' -->

# 9. Neural networks

### Representations, backpropagation and training control

---

<!-- _class: full-width objectives -->

## 9.1 Objectives

- Describe a **neuron** and a **multi-layer network**
- Explain **backpropagation** and gradient descent
- Name the usual **ingredients**
- Train a small network in **PyTorch**

---

<!-- _class: concept image-right -->

![neurone et activation](images/ml/illustrations/neurone-et-activation/portraits-travail--48-neurone-et-activation--9c57d907d551.jpg)

## 9.2 From a linear model to a neuron

$$a = \varphi\big(w^\top x + b\big)$$

- Weighted sum, then a **non-linear activation**
- **Logistic regression** = one neuron with a sigmoid

---

<!-- _class: concept image-right -->

![mlp](images/ml/illustrations/mlp/portraits-travail--49-mlp--33e5ec4e0474.jpg)

## 9.3 Multi-layer perceptron

- **Layers** of neurons, output of one = input of the next
- $a^{(1)} = \varphi(W^{(1)}x + b^{(1)})$, then $a^{(2)} = \varphi(W^{(2)}a^{(1)} + b^{(2)})$

> Fully connected, feed-forward

---

<!-- _class: concept image-right -->

![kernel intuition](images/ml/illustrations/kernel-intuition/deftech--20-kernel-intuition--adbf58e881cd.jpg)

## 9.4 Why non-linearity?

- Stacked **linear** layers = **one** linear layer
- The activation makes **curved**, complex functions possible
- **Universal approximation**: one wide hidden layer can approximate any continuous function

> It says a network **exists**, not that training will **find** it

![nonlinear activations](images/ml/schemas/nonlinear-activations/premieres-figures--09-02-nonlinear-activations--48cb52d63c90.png)

---

<!-- _class: concept image-right -->

![neurone et activation](images/ml/illustrations/neurone-et-activation/portraits-travail--48-neurone-et-activation--9c57d907d551.jpg)

## 9.5 Activations and outputs

| Function | Use |
|---|---|
| **ReLU** | Hidden layers (default) |
| **Sigmoid** | Binary probability |
| **Softmax** | Multi-class probabilities |
| **Identity** | Regression output |

- Regression: identity + **MSE**
- Binary: sigmoid + **cross-entropy**
- Multi-class: softmax + **cross-entropy**

---

<!-- _class: concept image-right -->

![retropropagation](images/ml/illustrations/backpropagation/recadrages-verticaux--25-backpropagation--e7a1798de2d0.jpg)

## 9.6 Training: backpropagation

1. **Forward pass**: compute activations, then the loss
2. **Backward pass**: propagate the error backwards, **chain rule**

- Cost of the same order as the forward pass
- Libraries do it automatically (`loss.backward()`)

![Diagram of backpropagation](images/ml/schemas/backpropagation/deftech--25-backpropagation--139adace76da.png)

> Proof in Annex H

---

<!-- _class: concept image-right -->

![retropropagation](images/ml/illustrations/retropropagation/portraits-travail--50-retropropagation--2a1d4b7ef6d0.jpg)

## 9.7 Backpropagation: one neuron by hand

- $x = 2$, $w = 0.3$, target $y = 1$, $\hat y = 0.6$, $L = 0.16$
- $\partial L/\partial w = 2(\hat y - y)\,x = -1.6$
- Update ($\eta = 0.1$): $w = 0.46$ → prediction **0.92**

> Closer to 1

---

<!-- _class: concept image-right -->

![optimization](images/ml/illustrations/optimization/deftech--08-optimization--d29c449ea175.jpg)

## 9.8 Optimisers

- **SGD**: basic mini-batch update
- **Momentum**: smooths and speeds up
- **Adam**: adaptive step per parameter, the **default** ($\eta \approx 10^{-3}$)
- **Epoch**: one pass over the training data



---

<!-- _class: concept image-right -->

![early stopping](images/ml/illustrations/optimization/deftech-v2--08-optimization--819baeb52821.jpg)

## 9.9 Training tricks

- **Initialisation** (He, Xavier)
- **Batch normalisation**
- **Dropout**: random neurons switched off
- **Weight decay**: like Ridge
- **Early stopping**

> The overfitting diagnosis is unchanged: training loss falls while validation loss rises.

---

<!-- _class: concept image-right -->

![vanishing and exploding gradients](images/ml/illustrations/vanishing-and-exploding-gradients/deftech--26-vanishing-and-exploding-gradients--428a202004c7.jpg)

## 9.10 Typical problems

- **Vanishing gradients**: ReLU, initialisation, normalisation
- **Exploding gradients**: gradient clipping
- **Overfitting**: dropout, weight decay, early stopping

![Diagram of vanishing and exploding gradients](images/ml/schemas/vanishing-and-exploding-gradients/deftech--26-vanishing-and-exploding-gradients--a44e3981be8d.png)

---

<!-- _class: concept image-right -->

![convolution](images/ml/illustrations/convolution/deftech--24-convolution--ef1e7394f37c.jpg)

## 9.11 Convolutional networks

- Small **filters** slide over the image and detect local patterns
- **Shared weights**: few parameters, same pattern anywhere
- **Pooling** shrinks the size
- Output size: $\lfloor (n + 2p - k)/s \rfloor + 1$

![convolution 3x3](images/ml/schemas/convolution-3x3/serie-2--08-convolution-3x3--5f9b51812acb.png)

>One output cell is a weighted sum of one input patch.
---

<!-- _class: concept image-right -->

![machine learning engineer](images/ml/illustrations/convolution/deftech-v2--24-convolution--92d5139c84ac.jpg)

## 9.12 A network in PyTorch

```python
loss = loss_fn(net(X_tr), y_tr)   # forward pass
opt.zero_grad()
loss.backward()                   # backward pass
opt.step()                        # gradient descent update
```

- **California Housing**: MLP for regression
- Compare with **gradient boosting**

---

<!-- _class: concept image-right -->

![decision trees](images/ml/illustrations/decision-trees/deftech--16-decision-trees--969967d242bc.jpg)

## 9.13 Question: who wins on tabular data?

- A neural network or gradient boosting on **California Housing**?

> Often a well-tuned tree ensemble, with much less tuning

---

<!-- _class: full-width accent-band -->

## 9. Exercises

📝 **9.1** Run one forward pass and one gradient update by hand
📝 **9.2** Explain why stacked linear layers remain a linear model
📝 **9.3** Compute convolution output sizes for two strides
📝 **9.4** Add dropout, plot train and validation RMSE, then compare with boosting

---

<!-- _class: full-width takeaway -->

## [9] — Key Takeaways

- Neural networks stack linear maps and non-linear activations to learn features
- Backpropagation computes the gradients used by gradient descent
- Validation, regularisation and early stopping control overfitting
- Gradient boosting remains a strong baseline for tabular data

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 10](images/generic/transitions-chapitre/piste-dessais/transitions-centrees--10-piste-dessais--5e68963ec9af.jpg)

---

<!-- _class: full-width -->
<!-- header: '[10] — Representations and Transfer Learning' -->

# 10. Representations and transfer learning

### Embeddings, pre-trained models and measured escalation

---

<!-- _class: full-width objectives -->

## 10.1 Objectives

- Explain what an **embedding** is
- Reuse a **pre-trained** model
- Place classical ML relative to **foundation models**

---

<!-- _class: concept image-right -->

![embeddings](images/ml/illustrations/embeddings/deftech--27-embeddings--ace4503931fe.jpg)

## 10.2 Representations and embeddings

- Models can **learn** the features
- An **embedding** is a vector encoding the **meaning** of an object
- Similar objects: **nearby** vectors
- **Cosine similarity**: $\cos(a,b) = \dfrac{a^\top b}{\lVert a\rVert\lVert b\rVert}$

![Diagram of embeddings](images/ml/schemas/embeddings/deftech--27-embeddings--300052878f18.png)

---

<!-- _class: concept image-right -->

![transfer learning](images/ml/illustrations/transfer-learning/deftech--28-transfer-learning--9ea134a20002.jpg)

## 10.3 Transfer learning

| Strategy | What you do |
|---|---|
| **Feature extraction** | Freeze the model, use its embeddings as features |
| **Fine-tuning** | Keep training some or all weights |

![Diagram of transfer learning](images/ml/schemas/transfer-learning/deftech--28-transfer-learning--820f26f3f9b6.png)

> Start with **feature extraction** when data and compute are limited.

---

<!-- _class: concept image-right -->

![transfer learning](images/ml/illustrations/transfer-learning/deftech--28-transfer-learning--9ea134a20002.jpg)

## 10.4 Risks of transfer learning

- **Domain mismatch**
- **Catastrophic forgetting** if fine-tuned too hard
- **Inherited biases**

---

<!-- _class: concept image-right -->

![machine learning engineer](images/generic/metiers/machine-learning-engineer/metiers--04-machine-learning-engineer--28b23eecc003.jpg)

## 10.5 Where classical ML ends

- **Foundation models**: pre-trained on huge corpora, covered in the GenAI course
- They **do not replace** classical ML on small, structured data
- Use the **simplest approach** that meets the requirement, and **measure**

---

<!-- _class: full-width accent-band -->

## 10. Exercises

📝 **10.1** Compute and interpret two cosine similarities
📝 **10.2** Choose a transfer strategy for 200 labelled images and no GPU
📝 **10.3** Extend the embedding example, cross-validate and inspect errors

---

<!-- _class: full-width takeaway -->

## [10] — Key Takeaways

- Useful embeddings place similar objects near one another
- Start with a frozen pre-trained model as a feature extractor
- Fine-tune only when measured gains justify the added cost
- Check domain mismatch, catastrophic forgetting and inherited bias

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 11](images/generic/transitions-chapitre/baie-de-calcul/transitions-centrees--11-baie-de-calcul--f49416172d0a.jpg)

---

<!-- _class: full-width -->
<!-- header: '[11] — Trustworthy Machine Learning' -->

# 11. Trustworthy machine learning

### Explainability, fairness, robustness and accountability

---

<!-- _class: full-width objectives -->

## 11.1 Objectives

- **Explain** a model, globally and per prediction
- Identify sources of **bias** and measure **fairness**
- Anticipate **robustness** problems
- **Document** a model

---

<!-- _class: concept image-right -->

![abstention](images/ml/illustrations/abstention/portraits-travail--57-abstention--03104b290586.jpg)

## 11.2 A good score is not trust

- **Understanding**
- **Fairness**
- **Robustness**
- **Privacy**
- **Accountability**

> High accuracy is not trustworthy

---

<!-- _class: concept image-right -->

![explainability](images/mlops/illustrations/explainability/deftech--29-explainability--62c6d814925e.jpg)

## 11.3 Explaining a model

| | Global | Local |
|---|---|---|
| **Question** | How does it behave overall? | Why this prediction? |
| **Tools** | Coefficients, **permutation importance**, PDP | **SHAP**, LIME |

- First ask: is an **interpretable model** accurate enough?

![Diagram of explainability](images/mlops/schemas/explainability/deftech--29-explainability--4d7a4c3e9594.png)

---

<!-- _class: concept image-right -->

![importance par permutation](images/ml/illustrations/importance-par-permutation/portraits-travail--43-importance-par-permutation--cd405ca1bf6d.jpg)

## 11.4 Permutation importance and SHAP

- **Permutation**: shuffle one feature on held-out data, measure the **score drop**
- **SHAP**: each feature gets a **contribution**; they add up to prediction minus average

> An explanation describes what the model **uses**, not a causal relation.

---

<!-- _class: concept image-right -->

![analyse d erreurs](images/ml/illustrations/analyse-d-erreurs/portraits-navals--53-analyse-d-erreurs--83ee03227b7e.jpg)

## 11.5 Sources of bias

- **Historical**: past decisions were discriminatory
- **Representation**: a group is under-represented
- **Measurement**: a proxy measures different things for different groups
- **Proxy variables**: postal code
- **Feedback loops**

---

<!-- _class: concept image-right -->

![equite et taux d erreur](images/ml/illustrations/equite-et-taux-d-erreur/portraits-travail--54-equite-et-taux-d-erreur--2bd1e2a65e01.jpg)

## 11.6 Fairness criteria

- **Demographic parity**: same rate of favourable decisions
- **Equal opportunity**: same **recall**
- **Equalised odds**: same true and false positive rates

> They generally **cannot all hold at once**

---

<!-- _class: concept image-right -->

![equite et taux d erreur](images/ml/illustrations/equite-et-taux-d-erreur/portraits-travail--54-equite-et-taux-d-erreur--2bd1e2a65e01.jpg)

## 11.7 Worked example: a loan model

- Approves **60 %** of group A, **30 %** of group B
- Demographic parity gap: **0.30**
- Among creditworthy applicants: 80 % vs 50 %, equal opportunity also **violated**

> Always measure metrics **per group**

![fairness error rates](images/ml/schemas/fairness-error-rates/premieres-figures--11-02-fairness-error-rates--2b5717d3a969.png)

---

<!-- _class: concept image-right -->

![equite et taux d erreur](images/ml/illustrations/equite-et-taux-d-erreur/portraits-travail--54-equite-et-taux-d-erreur--2bd1e2a65e01.jpg)

## 11.8 Debate: which criterion for Adult?

- **Adult Census**: income above 50 k$ as a "favourable decision"
- Measure rate and recall for `sex` and `race`
- Remove `sex`: does the gap close? Which **proxies** remain?

> No single right answer

---

<!-- _class: concept image-right -->

![robustness](images/mlops/illustrations/robustness/deftech--28-robustness--0ecf1c700d41.jpg)

## 11.9 Robustness

- **Distribution shift**: production differs from training
- **Out-of-distribution** inputs: a classifier always answers
- **Calibration**: "90 % sure" should be right 90 % of the time
- **Stress tests** on slices and degraded inputs

![Diagram of robustness](images/mlops/schemas/robustness/deftech--28-robustness--083b087de2bd.png)

---

<!-- _class: concept image-right -->

![data contracts](images/mlops/illustrations/data-contracts/deftech--09-data-contracts--9771ece71dca.jpg)

## 11.10 Privacy and regulation

- Models can **leak** training data
- Mitigations: data minimisation, access control, **differential privacy**
- **GDPR** for personal data
- **AI Act**: risk-based approach

> Check your application's category with your legal service

---

<!-- _class: concept image-right -->

![model card et limites](images/ml/illustrations/abstention/portraits-travail--57-abstention--03104b290586.jpg)

## 11.11 Documenting a model

- **Model card**: intended use, data, evaluation per group, limits
- **Datasheet** for the dataset: origin, composition, known biases

---

<!-- _class: full-width accent-band -->

## 11. Exercises

📝 **11.1** Diagnose a recall gap between demographic groups
📝 **11.2** Explain why deleting a sensitive feature does not remove its proxies
📝 **11.3** Design a stress test for a seasonal distribution shift
📝 **11.4** Compare permutation and impurity importance without claiming causality

---

<!-- _class: full-width takeaway -->

## [11] — Key Takeaways

- A good aggregate score is necessary but insufficient for trust
- Explanations describe model use, not causal relations
- Measure performance and errors for relevant groups
- Stress-test robustness and document intended use, limits and risks

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 12](images/generic/transitions-chapitre/satellite/transitions-centrees--12-satellite--3ceb2d5f2be5.jpg)

---

<!-- _class: full-width -->
<!-- header: '[12] — From Notebook to Product' -->

# 12. From notebook to product

### Reproducibility, packaging, serving and monitoring

---

<!-- _class: full-width objectives -->

## 12.1 Objectives

- Make an experiment **reproducible**
- **Package** a pipeline and **serve** it
- List what to **monitor**

---

<!-- _class: concept image-right -->

![git and dvc](images/mlops/illustrations/git-and-dvc/deftech--04-git-and-dvc--e7244570c8a1.jpg)

## 12.2 Reproducibility

Record exactly:

- the **code** version
- the **data** version
- the **environment**
- the **random seeds**
- the **hyperparameters** and **metrics**

![Diagram of git and dvc](images/mlops/schemas/git-and-dvc/deftech--04-git-and-dvc--ec90ff4251aa.png)

---

<!-- _class: concept image-right -->

![devops engineer](images/generic/metiers/devops-engineer/metiers--07-devops-engineer--5547a43661c5.jpg)

## 12.3 Code quality

- Notebook for **exploration**, **modules** for the real code
- **Tests** with `pytest`
- **Linter**, type hints, config file
- **Continuous integration**: tests at each commit

---

<!-- _class: concept image-right -->

![handoff et tracabilite](images/ml/illustrations/handoff-et-tracabilite/portraits-travail--59-handoff-et-tracabilite--54bfec45b5c7.jpg)

## 12.4 Package the whole pipeline

- Save **preprocessing + model**, not the model alone
- Otherwise: **training–serving skew**
- Load `joblib` / `pickle` files only from **trusted** sources

![Diagram of training serving skew](images/ml/schemas/training-serving-skew/deftech--30-training-serving-skew--9ba4e03d39d5.png)

---

<!-- _class: concept image-right -->

![inference api](images/mlops/illustrations/inference-api/deftech--14-inference-api--4d67a4f587ce.jpg)

## 12.5 Serve it behind an API

```python
@app.post("/predict")
def predict(sample: Sample):
    row = pd.DataFrame([sample.features])[art["columns"]]
    return {"probability": float(...)}
```

- **FastAPI**, one endpoint
- Return the **model version**

![Diagram of inference api](images/mlops/schemas/inference-api/deftech--14-inference-api--62aeac16e83a.png)

---

<!-- _class: concept image-right -->

![observability](images/mlops/illustrations/observability/deftech--18-observability--033d47c84b69.jpg)

## 12.6 Monitor in production

- **Data**: distributions, missing values, drift
- **Predictions**: output distribution, low confidence
- **Performance**: as soon as labels arrive
- **System**: latency, errors

![Diagram of observability](images/mlops/schemas/observability/deftech--18-observability--c1238c8fdf0c.png)

> Plan retraining and rollback before deployment.

---

<!-- _class: concept image-right -->

![machine learning engineer](images/mlops/illustrations/cicdct-and-quality-gates/deftech--11-cicdct-and-quality-gates--0e310002b92a.jpg)

## 12.7 What comes next

- **Machine Learning Engineer course**: MLOps, CI/CD for ML, deployment, edge

---

<!-- _class: full-width accent-band -->

## 12. Exercises

📝 **12.1** List what must be recorded to reproduce an experiment
📝 **12.2** Explain training-serving skew and how a pipeline prevents it
📝 **12.3** Run the API and test that its probability stays in $[0, 1]$

---

<!-- _class: full-width takeaway -->

## [12] — Key Takeaways

- Version code, data, environment, seeds, hyperparameters and metrics
- Package preprocessing and the model together to prevent training-serving skew
- Monitor data, predictions, performance and service health
- Plan retraining and rollback before deployment

---

<!-- _class: transition-chapitre -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

![bg cover Chapitre 13](images/generic/transitions-chapitre/bassin-houle/transitions-centrees--13-bassin-houle--c9b347e7117f.jpg)

---

<!-- _class: full-width -->
<!-- header: '[13] — Group Project' -->

# 13. Group project

### An evidence chain from problem to decision

---

<!-- _class: full-width takeaway -->

## 13.1 Group project — evidence chain

Your project must tell one coherent story:

**Problem** → **data** → **protocol** → **models** → **evidence** → **decision**

- Define the user, prediction target and **cost of errors**
- Build a leakage-safe pipeline and a **simple baseline**
- Compare at least two model families with the **same validation protocol**
- Analyse failures, one trust risk and the limits of the result

> Topic, team rules, deadlines and assessment grid: provided by the lecturer

---

<!-- _class: full-width takeaway -->

## 13.2 Project — definition of done

- A reproducible repository with environment and run instructions
- One notebook or report for exploration; reusable code for the pipeline
- A results table with metric, validation mean and spread
- One error analysis or slice analysis, not only a global score
- A short model card: intended use, data, limitations and monitoring signal
- A presentation that ends with a **decision**, not “model X scored 0.91”

---

<!-- _class: full-width takeaway -->

## [13] — Key Takeaways

- Connect the problem, data, protocol, models, evidence and decision
- Use a leakage-safe pipeline, a baseline and one validation protocol
- Support technical choices with reproducible evidence and error analysis
- Challenge fairness, deployment and monitoring assumptions
- Communicate a bounded, decision-ready result

---

<!-- _class: concept watermark takeaway -->
<!-- header: 'Machine Learning — Foundations and Practice' -->

![What to remember](images/generic/port-industriel/uuv-fusion-capteurs/port-industriel--02-uuv-fusion-capteurs--ce39a406c4e9.jpg)

## What to remember

- ML = rules **estimated from data**; what counts is **new** data
- **Split first**, pipeline everything, test set **once**
- **Baseline** before complexity
- **Bias–variance**: lowest validation error wins
- Metric = **cost of errors**
- **Trees** for tabular data, **scaling** for distances and gradients
- **Unsupervised** results need domain validation
- **Trust** means explaining, measuring per group and monitoring

---

<!-- _class: concept image-right -->

![data scientist](images/generic/decors-et-etudes/facade-portuaire/versions-isolees--facade-portuaire--f5868bc079bb.jpg)

## Resources

- Bishop, C. M. (2006). Pattern recognition and machine learning. Springer
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An introduction to statistical learning: With applications in Python. Springer. https://doi.org/10.1007/978-3-031-38747-0
- Murphy, K. P. (2022). Probabilistic machine learning: An introduction. MIT Press
- Shalev-Shwartz, S., & Ben-David, S. (2014). Understanding machine learning: From theory to algorithms Cambridge University Press. https://doi.org/10.1017/CBO9781107298019
- Géron, A. (2025). Hands-on machine learning with Scikit-Learn and PyTorch: Concepts, tools, and techniques to build intelligent systems. O’Reilly Media
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079–2107
- Molnar, C. (2025). Interpretable machine learning: A guide for making black box models explainable (3rd ed.)
- scikit-learn documentation
- PyTorch documentation
- Hugging Face courses and Hub


---

<!-- _class: full-width -->
<!-- _paginate: false -->

# Questions?
