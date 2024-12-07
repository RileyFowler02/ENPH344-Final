import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def state_preparation(state):
    qml.StatePrep(state, wires=["S"])

def entangle_qubits():
    qml.Hadamard(wires="A")
    qml.CNOT(wires=["A", "B"])

def basis_rotation():
    qml.CNOT(wires=["S", "A"])
    qml.Hadamard(wires="S")

def measure_and_update():
    m0 = qml.measure("S")
    m1 = qml.measure("A")
    qml.cond(m1, qml.PauliX)("B")
    qml.cond(m0, qml.PauliZ)("B")

def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()

state = np.array([1 / np.sqrt(2) + 0.3j, 0.4 - 0.5j])
dev = qml.device("default.qubit", wires=["S", "A", "B"])

@qml.qnode(dev)
def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()
    return qml.density_matrix(wires=["B"])

# Draw the circuit
fig, ax = qml.draw_mpl(teleport, style="pennylane", level="device")(state)
plt.show()