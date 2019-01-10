from base.base_model import BaseModel
import tensorflow as tf
from configs.config import config, layers_details

class MNIST_Model(BaseModel):
    def __init__(self, config, layer_details):
        super(MNIST_Model, self).__init__(config)
        self.layer_details = layer_details
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.

        self.learning_rate = tf.placeholder(tf.float32, shape=(),
                                            name='learning_rate')

        self.X = tf.placeholder(tf.float32,
                                shape=[self.config["batch_size"]] + list(
                                    self.config["input_shape"]),
                                name='input_image_vector')
        self.Y = tf.placeholder(tf.float32,
                                shape=[self.config["batch_size"]] + list(
                                    self.config["output_shape"]),
                                name='output_layer')
        layer_1_FC = tf.contrib.layers.fully_connected(self.X,
                                                       self.layer_details[
                                                           'FullyConnected_1'])
        layer_2_FC = tf.contrib.layers.fully_connected(layer_1_FC,
                                                       self.layer_details[
                                                           'FullyConnected_2'])
        layer_3_FC = tf.contrib.layers.fully_connected(layer_2_FC,
                                                       self.layer_details[
                                                           'FullyConnected_3'])

        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass


if __name__ == '__main__':
    print ("Validating Model")
    mnist_model = MNIST_Model(config, layers_details)
