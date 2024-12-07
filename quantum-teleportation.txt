import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Function to prepare the initial state
def state_preparation(state):
    qml.StatePrep(state, wires=["S"])

# Function to entangle the qubits
def entangle_qubits():
    qml.Hadamard(wires="A")
    qml.CNOT(wires=["A", "B"])

# Function to perform basis rotation
def basis_rotation():
    qml.CNOT(wires=["S", "A"])
    qml.Hadamard(wires=["S"])

# Function to measure and update the state
def measure_and_update():
    m0 = qml.measure("S")
    m1 = qml.measure("A")
    qml.cond(m1, qml.PauliX)("B")
    qml.cond(m0, qml.PauliZ)("B")

# Function to perform the teleportation protocol
def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()

# Define the initial state to be teleported
state = np.array([1 / np.sqrt(2) + 0.3j, 0.4 - 0.5j])

# Create a device with 3 qubits
dev = qml.device("default.mixed", wires=["S", "A", "B"])

# Define a QNode for the teleportation protocol
@qml.qnode(dev)
def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()
    return qml.density_matrix(wires=["B"])

# Define a QNode for the teleportation protocol with noise
@qml.qnode(dev)
def noisy_teleport(state, p):
    state_preparation(state)
    entangle_qubits()
    qml.DepolarizingChannel(p, wires="A")
    qml.DepolarizingChannel(p, wires="B")
    basis_rotation()
    measure_and_update()
    return qml.density_matrix(wires=["B"])

# Function to calculate the fidelity
def fidelity(state, p):
    original_density_matrix = qml.math.dm_from_state_vector(state)
    teleported_density_matrix = teleport(state)
    noisy_teleported_density_matrix = noisy_teleport(state, p)

    fidelity_clean = qml.math.fidelity(original_density_matrix, teleported_density_matrix)
    fidelity_noisy = qml.math.fidelity(original_density_matrix, noisy_teleported_density_matrix)

    return fidelity_clean, fidelity_noisy

# Range of p values for depolarizing noise
p_values = np.linspace(0, 1, 100)
fidelities_clean = []
fidelities_noisy = []

# Calculate the fidelity for each p value
for p in p_values:
    fidelity_clean, fidelity_noisy = fidelity(state, p)
    fidelities_clean.append(fidelity_clean)
    fidelities_noisy.append(fidelity_noisy)

# Plot the fidelities
plt.plot(p_values, fidelities_clean, label='Clean')
plt.plot(p_values, fidelities_noisy, label='Noisy')
plt.xlabel('Depolarizing Probability (p)')
plt.ylabel('Fidelity')
plt.title('Fidelity of Teleported State vs. Depolarizing Noise')
plt.legend()
plt.show()

# Find the slope of the noisy data from p=0 to p=0.5
p_values_half = p_values[p_values <= 0.5]
fidelities_noisy_half = fidelities_noisy[:len(p_values_half)]

slope, intercept, r_value, p_value, std_err = linregress(p_values_half, fidelities_noisy_half)
print(f"Slope of the noisy data from p=0 to p=0.5: {slope}")

# Print the y-intercept
print(f"Y-intercept of the noisy data: {intercept}")

# Equation of the line
print(f"Equation of the line: y = {slope}p + {intercept}")