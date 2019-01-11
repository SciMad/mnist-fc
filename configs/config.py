''' A file containing a config class and the associated hyperparameters'''
NUM_CLASSES = 1000
config = {
            "mnist-data": "../data/mnist.pkl",
            "batch_size": 1,
            "learning_rate": 0.01,
            "input_shape": (None, 784, 1),
            "output_shape": (None, NUM_CLASSES),
            "num_epoch" : 400,
            "checkpoint_path": "../data/output/",
            "max_to_keep": 100,
            "num_classes": 10,
            "steps": 100
            }


layers_details = {
            "FullyConnected_1": 100,
            "FullyConnected_2": 30,
            "FullyConnected_3": config['num_classes']
            }
