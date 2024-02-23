# Neural-Network
Author : Fauz Ahmed

Developed a neural network architecture using separate processes and threads on a multi-core processor.
The system uses inter-process communication through pipes for exchanging information such as weights and biases between processes.
Each layer of the neural network is represented as a separate process, and each neuron within a layer is treated as a separate thread.
During backpropagation, the error signal is propagated backward through the layers of the network,
and the system updates the weights and biases based on the calculated gradients, while utilizing the processing power of multiple cores.
