import pennylane as qml
from pennylane import numpy as np

# Define a device with 1 qubit
dev = qml.device("default.qubit", wires=1)

# Define a quantum node (quantum function)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RZ(params[2], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the parameters
params = np.array([0.1, 0.2, 0.3], requires_grad=True)

# Execute the circuit
result = circuit(params)

print(f"Expectation value: {result}")