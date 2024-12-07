import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the squeezing parameter
r = np.arcsinh(1 / 2)

# Calculate the beamsplitter transmittance
T = 1 / 4 * (1 - np.tanh(r))

# Define the CV CNOT gate
def cv_cnot():
    qml.Squeezing(r, 0, wires=0)  # Squeeze the control qumode
    qml.Squeezing(r, 0, wires=1)  # Squeeze the target qumode
    qml.Beamsplitter(np.arccos(np.sqrt(T)), 0, wires=[0, 1])  # Apply beamsplitter
    qml.Squeezing(-r, 0, wires=0)  # Unsqueeze the control qumode
    qml.Squeezing(-r, 0, wires=1)  # Unsqueeze the target qumode

# Create a device with 2 qumodes
dev = qml.device("default.gaussian", wires=2)

# Define separate QNodes for qumode 0 and qumode 1 (pre-CNOT)
@qml.qnode(dev)
def pre_cnot_quadrature_qmode_0():
    qml.CoherentState(1, 0, wires=0)
    qml.CoherentState(1, np.pi / 2, wires=1)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev)
def pre_cnot_quadrature_qmode_1():
    qml.CoherentState(1, 0, wires=0)
    qml.CoherentState(1, np.pi / 2, wires=1)
    return qml.expval(qml.QuadX(1))

# Define separate QNodes for qumode 0 and qumode 1 (post-CNOT)
@qml.qnode(dev)
def post_cnot_quadrature_qmode_0():
    qml.CoherentState(1, 0, wires=0)
    qml.CoherentState(1, np.pi / 2, wires=1)
    cv_cnot()
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev)
def post_cnot_quadrature_qmode_1():
    qml.CoherentState(1, 0, wires=0)
    qml.CoherentState(1, np.pi / 2, wires=1)
    cv_cnot()
    return qml.expval(qml.QuadX(1))

# Calculate X-quadrature samples
samples_pre_0 = [pre_cnot_quadrature_qmode_0() for _ in range(100)]
samples_pre_1 = [pre_cnot_quadrature_qmode_1() for _ in range(100)]
samples_post_0 = [post_cnot_quadrature_qmode_0() for _ in range(100)]
samples_post_1 = [post_cnot_quadrature_qmode_1() for _ in range(100)]

# Plot the X-quadrature distributions (using histograms)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(samples_pre_0, bins=20)
axes[0, 0].set_title("X-quadrature samples (Pre-CNOT, Qmode 0)")
axes[0, 1].hist(samples_pre_1, bins=20)
axes[0, 1].set_title("X-quadrature samples (Pre-CNOT, Qmode 1)")
axes[1, 0].hist(samples_post_0, bins=20)
axes[1, 0].set_title("X-quadrature samples (Post-CNOT, Qmode 0)")
axes[1, 1].hist(samples_post_1, bins=20)
axes[1, 1].set_title("X-quadrature samples (Post-CNOT, Qmode 1)")
plt.tight_layout()
plt.show()