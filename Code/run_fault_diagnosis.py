"""Bearing fault diagnosis: quantum (QSVM, VQC, QNN, HQCNN) vs. classical models.

    python run_fault_diagnosis.py --data ../Dataset/CWRU          # real CWRU data
    python run_fault_diagnosis.py --synthetic                     # pipeline smoke test only
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from classical_models import classifiers
from data import load_cwru, synthetic_cwru
from features import angle_encoder, extract_features
from quantum_models import HQCNN, AmplitudeQSVC, make_qnn_classifier, make_qsvc, make_vqc


def train_hqcnn(Xtr, ytr, Xte, n_qubits, n_classes, epochs):
    torch.manual_seed(0)
    model = HQCNN(n_qubits, n_classes)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    xt, yt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor(ytr)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt)).split(32):
            opt.zero_grad()
            nn_loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            nn_loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CWRU root with Normal/Ball/InnerRace/OuterRace sub-folders")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=400, help="cap on training samples for quantum models")
    ap.add_argument("--hqcnn-epochs", type=int, default=15)
    ap.add_argument("--out", default="../Results/fault_diagnosis.json")
    args = ap.parse_args()

    X, y, classes = synthetic_cwru() if args.synthetic else load_cwru(args.data)
    F = extract_features(X)
    Xtr, Xte, Ftr, Fte, ytr, yte = train_test_split(X, F, y, test_size=0.3, stratify=y, random_state=0)
    if len(ytr) > args.max_train:
        keep = np.random.default_rng(0).choice(len(ytr), args.max_train, replace=False)
        Xtr, Ftr, ytr = Xtr[keep], Ftr[keep], ytr[keep]
    enc = angle_encoder(args.qubits).fit(Ftr)
    Qtr, Qte = enc.transform(Ftr), enc.transform(Fte)
    Str, Ste = enc.named_steps["std"].transform(Ftr), enc.named_steps["std"].transform(Fte)
    # amplitude encoding: all 13 features (+ offset) -> 16 amplitudes of 4 qubits
    Atr, Ate = (np.column_stack([S, np.ones(len(S))]) for S in (Str, Ste))

    models = {name: ("classical", m) for name, m in classifiers().items()}
    models |= {"QSVM": ("quantum", make_qsvc(args.qubits)),
               "VQC": ("quantum", make_vqc(args.qubits, len(classes))),
               "QNN": ("quantum", make_qnn_classifier(args.qubits, len(classes))),
               "QSVM (amplitude)": ("amplitude", AmplitudeQSVC())}

    results = {}
    for name, (kind, model) in models.items():
        t0 = time.time()
        a, b = {"quantum": (Qtr, Qte), "amplitude": (Atr, Ate)}.get(kind, (Str, Ste))
        model.fit(a, ytr)
        pred = np.asarray(model.predict(b)).ravel()
        results[name] = {"accuracy": accuracy_score(yte, pred), "f1_macro": f1_score(yte, pred, average="macro"),
                         "train_accuracy": accuracy_score(ytr, np.asarray(model.predict(a)).ravel()),
                         "seconds": time.time() - t0}
        print(f"{name:14s} acc {results[name]['accuracy']:.3f}  F1 {results[name]['f1_macro']:.3f}  ({results[name]['seconds']:.1f}s)")

    t0 = time.time()
    pred = train_hqcnn(Xtr, ytr, Xte, args.qubits, len(classes), args.hqcnn_epochs)
    results["HQCNN"] = {"accuracy": accuracy_score(yte, pred), "f1_macro": f1_score(yte, pred, average="macro"),
                        "seconds": time.time() - t0}
    print(f"{'HQCNN':14s} acc {results['HQCNN']['accuracy']:.3f}  F1 {results['HQCNN']['f1_macro']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"synthetic": args.synthetic, "classes": classes, "qubits": args.qubits,
                                          "n_train": len(ytr), "n_test": len(yte), "results": results}, indent=2))


if __name__ == "__main__":
    main()
