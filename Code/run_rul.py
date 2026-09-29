"""Remaining-useful-life (RUL) estimation: QSVR and HQCNN vs. classical regressors.

    python run_rul.py --train ../Dataset/XJTU-SY/35Hz12kN/Bearing1_1 ../Dataset/XJTU-SY/35Hz12kN/Bearing1_2 \
                      --test  ../Dataset/XJTU-SY/35Hz12kN/Bearing1_3
    python run_rul.py --synthetic                                   # pipeline smoke test only
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from classical_models import regressors
from data import load_xjtu_bearing, synthetic_run_to_failure
from features import angle_encoder, extract_features
from quantum_models import HQCNN, make_qsvr


def load(folders, synthetic, seeds):
    parts = [synthetic_run_to_failure(seed=s) for s in seeds] if synthetic else [load_xjtu_bearing(f) for f in folders]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train_hqcnn_regressor(Xtr, ytr, Xte, n_qubits, epochs):
    torch.manual_seed(0)
    model = HQCNN(n_qubits, regression=True)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    xt, yt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt)).split(32):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--fs", type=int, default=25600, help="sampling rate (XJTU-SY: 25.6 kHz)")
    ap.add_argument("--hqcnn-epochs", type=int, default=20)
    ap.add_argument("--out", default="../Results/rul.json")
    args = ap.parse_args()

    Xtr, ytr = load(args.train, args.synthetic, seeds=[0, 1])
    Xte, yte = load(args.test, args.synthetic, seeds=[2])
    Ftr, Fte = extract_features(Xtr, args.fs), extract_features(Xte, args.fs)
    enc = angle_encoder(args.qubits).fit(Ftr)

    preds = {}
    for name, m in regressors().items():
        preds[name] = m.fit(enc.named_steps["std"].transform(Ftr), ytr).predict(enc.named_steps["std"].transform(Fte))
    preds["QSVR"] = make_qsvr(args.qubits).fit(enc.transform(Ftr), ytr).predict(enc.transform(Fte))
    preds["HQCNN"] = train_hqcnn_regressor(Xtr, ytr, Xte, args.qubits, args.hqcnn_epochs)

    results = {}
    for name, p in preds.items():
        p = np.clip(p, 0, 1)
        results[name] = {"rmse": float(np.sqrt(mean_squared_error(yte, p))), "mae": float(mean_absolute_error(yte, p))}
        print(f"{name:14s} RMSE {results[name]['rmse']:.3f}  MAE {results[name]['mae']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"synthetic": args.synthetic, "qubits": args.qubits, "results": results}, indent=2))
    if not args.synthetic:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(yte, "k-", lw=2, label="True RUL")
        for name in ("QSVR", "HQCNN", "SVR (RBF)"):
            ax.plot(np.clip(preds[name], 0, 1), label=name, alpha=0.8)
        ax.set_xlabel("snapshot"); ax.set_ylabel("RUL (fraction of life)"); ax.legend()
        fig.tight_layout(); fig.savefig("../Figures/rul_prediction.png", dpi=200)


if __name__ == "__main__":
    main()
