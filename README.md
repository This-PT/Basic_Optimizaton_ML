# Bank Marketing: Who Should We Call?

This is my traditional machine learning project using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing). I wanted to build more than a model that prints an accuracy number. The goal is to understand the full workflow: define when a prediction happens, prepare the data, compare models fairly, choose a decision threshold, and inspect mistakes.

## The question

**Before deciding which customers to call, can we identify customers more likely to subscribe to a term deposit?**

The dataset has 45,211 customer records. About 11.7% have `y = yes`, so the classes are imbalanced. A model that always predicts `no` gets about 88.3% accuracy but finds no subscribers. That is why I focus on precision, recall, F1, balanced accuracy, and the confusion matrix instead of accuracy alone.

## Features available at prediction time

The model uses 11 features. I removed `y` because it is the answer. I also removed `duration`, `contact`, `day`, `month`, and `campaign` from the inputs for this project's **before-calling** decision point. They describe the current contact or campaign and would not necessarily be known when choosing whom to call. This timing assumption matters: a different prediction point could allow a different feature set.

I kept features such as `age`, `balance`, `housing`, `loan`, `pdays`, `previous`, and `poutcome`. The last three describe *previous* campaign history, so they can be available before the current campaign. A previous `success` is a clue, not the answer to whether a customer subscribes this time.

## Method

1. Split the data into 80% development data (`X_train`) and 20% held-out test data, stratified by `y`.
2. Put numeric imputation and scaling, plus categorical imputation and one-hot encoding, inside each model pipeline. This lets every cross-validation fold learn preprocessing only from its training rows.
3. Compare a dummy model, balanced logistic regression, and a balanced decision tree.
4. Use five stratified folds within `X_train` to obtain out-of-fold predictions. Every development row is predicted by a model that did not train on that row.
5. Compare thresholds `0.30`, `0.40`, `0.50`, `0.60`, and `0.70` using the `yes` class F1 score. Refit the selected model on all of `X_train`, then evaluate it on the test data.

I also tried a small tree grid in `tune_tree.py`: depths 4, 6, and 8, with minimum leaf sizes 50 and 100. The depth-6, leaf-50 tree had the highest out-of-fold F1 among these tested combinations. The differences from some other settings were small, so I do not treat it as a guaranteed optimum.

## Results

| Model | Selected threshold | Out-of-fold yes F1 | Out-of-fold balanced accuracy |
| --- | ---: | ---: | ---: |
| Dummy (always `no`) | — | 0.000 | 0.500 |
| Logistic regression | 0.60 | 0.354 | 0.642 |
| Decision tree (`max_depth=6`, `min_samples_leaf=50`) | 0.70 | **0.392** | **0.655** |

The decision tree won among the tested models by out-of-fold `yes` F1. Threshold `0.70` was only slightly ahead of `0.60` (`0.392` versus `0.390`), so I see them as practically close rather than claiming that `0.70` is universally best.

For the selected tree on the test data:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.860 |
| Balanced accuracy | 0.656 |
| `yes` precision | 0.400 |
| `yes` recall | 0.390 |
| `yes` F1 | 0.395 |

| Actual / predicted | Predicted `no` | Predicted `yes` |
| --- | ---: | ---: |
| Actual `no` | 7,366 | 619 |
| Actual `yes` | 645 | 413 |

The tree identified 413 of the 1,058 subscribers in the test set. Of the 1,032 customers it predicted as `yes`, 413 subscribed. This is a prioritization signal, not proof that calling these customers would be profitable; the dataset does not provide call costs or subscription value.

## What the errors taught me

`poutcome` means the outcome of a **previous** marketing campaign:

- `success`: 306 test customers had previous success. The final tree predicted `yes` for all 306; 199 subscribed this time and 107 did not.
- `unknown`: 7,370 test customers had no known previous outcome. The tree missed 548 of the 657 actual subscribers in this group.

The final tree's split-based feature importance was highest for `poutcome_success` (0.462), followed by `housing_no` (0.183), `age` (0.137), `balance` (0.097), and `pdays` (0.084). These numbers describe how much the tree used each feature for splits. They do **not** show a causal effect or a percentage change in subscription probability.

## How to run

From this folder, with the dataset file `bank-full.csv` present:

```bat
python -m pip install numpy pandas scikit-learn
python baseline.py
```

`baseline.py` prints the cross-validation comparison, selected model and threshold, test metrics, confusion matrix, errors by previous campaign outcome, and tree feature importance.

To repeat the separate tree-setting comparison:

```bat
python tune_tree.py
```

## Limitations and next steps

This is a learning experiment. I inspected the test results during earlier development before the final cross-validation version, so the test set is **not a pristine one-time final holdout**. I report its score as an exploratory result, not an unbiased final performance claim. For a stronger future evaluation, I would set aside a fresh untouched test set before making further modeling decisions. I would also check how performance changes over time and whether predicted scores need calibration before using them as probabilities in a real decision process.

"# Basic_Optimizaton_ML" 
