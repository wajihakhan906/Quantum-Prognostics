# Results

## Published: Bearing Fault Diagnosis Using Quantum Machine Learning (IEEE)
Selected classical vibration features are amplitude-encoded into qubits and classified with a
**quantum support vector machine** (simulated on IBM Quantum resources):

| Metric | QSVM |
|---|---|
| Training accuracy | **99.61 %** |
| Testing accuracy | **99.14 %** |
| F1-score / Recall / Precision | **99 %** |

The QSVM outperformed kernel extreme learning machine (KELM), extreme learning machine (ELM), SVM and Softmax
classifiers. Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10712885/).

## Reproducing with this code
```bash
cd Code
python run_fault_diagnosis.py --data ../Dataset/CWRU        # writes Results/fault_diagnosis.json
python run_rul.py --train <bearing folders> --test <bearing folder>   # writes Results/rul.json, Figures/rul_prediction.png
```
Each JSON records accuracy / macro-F1 (classification) or RMSE / MAE (RUL) for every quantum, hybrid and classical model.
