"""Quantum and hybrid models: QSVM, VQC, QNN, QSVR and HQCNN."""
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import QSVC, QSVR, VQC, NeuralNetworkClassifier
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA


def feature_map(n_qubits, reps=2):
    return zz_feature_map(n_qubits, reps=reps, entanglement="linear")


def ansatz(n_qubits, reps=2):
    return real_amplitudes(n_qubits, reps=reps, entanglement="linear")


def quantum_kernel(n_qubits, reps=2):
    return FidelityStatevectorKernel(feature_map=feature_map(n_qubits, reps))


def make_qsvc(n_qubits, C=1.0):
    return QSVC(quantum_kernel=quantum_kernel(n_qubits), C=C)


def make_qsvr(n_qubits, C=1.0, epsilon=0.05):
    return QSVR(quantum_kernel=quantum_kernel(n_qubits), C=C, epsilon=epsilon)


def make_vqc(n_qubits, n_classes, maxiter=100):
    return VQC(feature_map=feature_map(n_qubits, 1), ansatz=ansatz(n_qubits), optimizer=COBYLA(maxiter=maxiter),
               sampler=StatevectorSampler(), loss="cross_entropy")


def _qnn(n_qubits, n_outputs):
    fm, an = feature_map(n_qubits, 1), ansatz(n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(an, inplace=True)
    obs = [SparsePauliOp("I" * (n_qubits - 1 - i) + "Z" + "I" * i) for i in range(n_outputs)]
    return EstimatorQNN(circuit=qc, observables=obs, input_params=fm.parameters, weight_params=an.parameters,
                        estimator=StatevectorEstimator())


def make_qnn_classifier(n_qubits, n_classes, maxiter=100):
    """EstimatorQNN with one <Z_i> readout per class, trained with cross-entropy."""
    return NeuralNetworkClassifier(_qnn(n_qubits, n_classes), optimizer=COBYLA(maxiter=maxiter),
                                   loss="cross_entropy", one_hot=True)


class HQCNN(nn.Module):
    """Hybrid quantum-classical CNN: 1D CNN on raw vibration -> n angles -> QNN -> linear head."""

    def __init__(self, n_qubits=4, n_outputs=4, regression=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, 64, stride=8, padding=28), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(8, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(), nn.Linear(128, n_qubits), nn.Tanh(),
        )
        self.quantum = TorchConnector(_qnn(n_qubits, n_qubits),
                                      initial_weights=0.1 * np.random.default_rng(0).standard_normal(
                                          len(ansatz(n_qubits).parameters)))
        self.head = nn.Linear(n_qubits, 1 if regression else n_outputs)
        self.regression = regression

    def forward(self, x):
        angles = (self.cnn(x.unsqueeze(1)) + 1) * (np.pi / 2)  # [-1, 1] -> [0, pi]
        out = self.head(self.quantum(angles))
        return torch.sigmoid(out).squeeze(-1) if self.regression else out


# --------------------------------------------------------------------------- #
# Amplitude-encoded QSVM (features stored in the 2^n amplitudes of n qubits)
# --------------------------------------------------------------------------- #
def amplitude_state_circuit(x):
    """State-preparation circuit |x> = sum_i x_i/||x|| |i> (pads to the next power of two)."""
    from qiskit.circuit.library import StatePreparation

    dim = 1 << max(1, int(np.ceil(np.log2(len(x)))))
    v = np.zeros(dim)
    v[: len(x)] = x
    v = v / np.linalg.norm(v)
    qc = QuantumCircuit(int(np.log2(dim)))
    qc.append(StatePreparation(v), qc.qubits)
    return qc


class AmplitudeQSVC:
    """QSVM with amplitude encoding. The statevector fidelity kernel |<x|x'>|^2 of two
    amplitude-encoded states equals (x . x' / (||x|| ||x'||))^2, which is evaluated exactly here;
    on hardware the same kernel is estimated with `amplitude_state_circuit` + a swap/compute-uncompute test.
    """

    def __init__(self, C=10.0):
        from sklearn.svm import SVC

        self.svc = SVC(kernel="precomputed", C=C)

    @staticmethod
    def _unit(X):
        X = np.asarray(X, dtype=float)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def kernel(self, A, B):
        return (self._unit(A) @ self._unit(B).T) ** 2

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.svc.fit(self.kernel(self.X_, self.X_), y)
        return self

    def predict(self, X):
        return self.svc.predict(self.kernel(X, self.X_))
