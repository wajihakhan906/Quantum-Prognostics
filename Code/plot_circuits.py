"""Renders the feature map, ansatz and QNN circuits to ../Figures."""
from qiskit import QuantumCircuit

from quantum_models import ansatz, feature_map


def main(n_qubits=4):
    fm, an = feature_map(n_qubits), ansatz(n_qubits)
    fm.draw("mpl", filename="../Figures/zz_feature_map.png", fold=30)
    an.draw("mpl", filename="../Figures/real_amplitudes_ansatz.png", fold=30)
    qnn = QuantumCircuit(n_qubits)
    qnn.compose(feature_map(n_qubits, 1), inplace=True)
    qnn.barrier()
    qnn.compose(an, inplace=True)
    qnn.draw("mpl", filename="../Figures/qnn_circuit.png", fold=40)


if __name__ == "__main__":
    main()
