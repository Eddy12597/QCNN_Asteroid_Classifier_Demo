
import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from PIL import Image



estimator = Estimator()

def load_dataset(data_dir='data',max_samples_per_class = None):
    images = []
    labels = []

    class_mapping = {
        'asteroids': 1,
        'no_asteroids': -1
    }

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        count = 0

        for filename in os.listdir(class_dir):

            if max_samples_per_class is not None and count >= max_samples_per_class:
                break

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:

                    img_path = os.path.join(class_dir, filename)
                    img = Image.open(img_path).convert('L')
                    images.append(img)
                    labels.append(label)
                    count += 1
                except Exception:
                    continue

    return images, labels


obj_func_vals = []
def my_callback(weight, obj_func_eval):
    obj_func_vals.append(obj_func_eval)

    if len(obj_func_vals) % 10 == 0:
        print(len(obj_func_vals), "obj_func_eval", obj_func_eval)


def preprocess_image(img, size=(2,2)):
    img = img.convert('L')
    img_resized = img.resize(size, Image.BILINEAR)
    arr = np.array(img_resized) / 255.0
    max_val = np.max(arr)
    arr = arr/max_val
    angles = arr * np.pi/2
    return angles.flatten()


def conv_circuit(params):
    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi/2,1)
    circuit.cx(1,0)
    circuit.rz(params[0],0)
    circuit.ry(params[1],1)
    circuit.cx(0,1)
    circuit.rz(params[2],1)
    circuit.cx(1,0)
    circuit.rz(np.pi/2,0)

    return circuit


def pool_circuit(params):
    circuit = QuantumCircuit(2)
    circuit.rz(-np.pi/2,1)
    circuit.cx(1,0)
    circuit.rz(params[0],0)
    circuit.ry(params[1],1)
    circuit.cx(0,1)
    circuit.ry(params[2],1)

    return circuit


def conv_layer(num_qubit,param_prefix):
    qc = QuantumCircuit(num_qubit, name = "Conv")
    qubits = list(range(num_qubit))
    params = ParameterVector(param_prefix, length=num_qubit * 3)

    param_i = 0
    #even pair operations
    for q1,q2 in zip(qubits[0::2],qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_i:(param_i+3)]),[q1,q2])
        qc.barrier()
        param_i += 3
    #odd pair operations

    for q1,q2 in zip(qubits[1::2],qubits[2::2]+[0]):
        qc = qc.compose(conv_circuit(params[param_i:(param_i + 3)]), [q1, q2])
        qc.barrier()
        param_i += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubit)
    qc.append(qc_inst,qubits)

    return qc


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name = "Pool")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits//2 * 3)

    for sources, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:(param_index + 3)]), [sources, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst,range(num_qubits))
    return qc


def main(max_samples_per_class = 131):
    images, labels = load_dataset(data_dir='data', max_samples_per_class=max_samples_per_class)
    images = [preprocess_image(img) for img in images]


    train_images, test_images, train_labels, test_labels = train_test_split(
images, labels, test_size=0.3, random_state=246
    )

    print(len(images))



    """train model"""
    featuremap = ZFeatureMap(4)

    qcnn = QuantumCircuit(4, name = "qcnn")
    #First Layer
    qcnn = qcnn.compose(conv_layer(4, "c1"), [0,1,2,3])
    qcnn = qcnn.compose(pool_layer([0, 1], [2, 3], "p1"), [0,1,2,3])
    #Second Layer
    qcnn = qcnn.compose(conv_layer(2, "c2"), [2,3])
    qcnn = qcnn.compose(pool_layer([0], [1], "p2"), [2,3])

    circuit = QuantumCircuit(4)
    circuit.compose(featuremap,range(4), inplace = True)
    circuit.compose(qcnn, range(4), inplace = True)
    # circuit.draw("mpl")
    # plt.savefig("plot.png")
    # circuit.decompose().draw(output='mpl', filename='circuit.png')
    # plt.savefig("plot2.png")
    # plt.show()

    observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=featuremap.parameters,
        weight_params=qcnn.parameters,
        estimator=estimator
    )


    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=100),
        callback = my_callback
    )

    train_x = np.asarray(train_images)
    train_y = np.asarray(train_labels)
    classifier.fit(train_x, train_y)

    y_predict = classifier.predict(test_images)
    test_x = np.asarray(test_images)
    test_y = np.asarray(test_labels)

    accuracy = np.round(100 * classifier.score(test_x, test_y), 2)
    print(f"Accuracy from the test data : {accuracy}%")

    # y = obj_func_vals
    # x = range(len(y))
    #
    # plt.plot(x, y)
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective Function")
    # plt.title("Objective Function Value vs Iteration")
    # plt.savefig("objective_function.png")
    # plt.show()


    return accuracy


if __name__ == "__main__":
    main()






