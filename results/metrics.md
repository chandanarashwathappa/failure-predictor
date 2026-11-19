| Model | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | Missed failures (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|
| Always 'no failure' | 0.50 | 0.966 | 0.000 | 0.000 | 0.000 | 0.034 | 68 | 0 |
| Logistic regression (raw sensors) | 0.88 | 0.958 | 0.375 | 0.353 | 0.364 | 0.382 | 44 | 40 |
| Logistic regression (+ physics features) | 0.92 | 0.968 | 0.532 | 0.368 | 0.435 | 0.430 | 43 | 22 |
| Random forest (raw sensors) | 0.35 | 0.979 | 0.658 | 0.765 | 0.707 | 0.755 | 16 | 27 |
| Random forest (+ physics features) | 0.44 | 0.993 | 0.949 | 0.824 | 0.882 | 0.878 | 12 | 3 |

Best model (PR-AUC): **Random forest (+ physics features)**

| Failure mode | Failures in test set | Caught |
|---|---|---|
| TWF | 10 | 2 |
| HDF | 29 | 28 |
| PWF | 13 | 12 |
| OSF | 16 | 16 |
| RNF | 0 | 0 |
