from base.base_train import BaseTrain
import tensorflow as tf
from models.model import MNIST_Model
from dataset.dataset import DataGenerator
from configs.config import config, layers_details


class MNISTTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(MNISTTrainer, self).__init__(sess, model, data, config, logger)


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass


if __name__ == '__main__':

    model = MNIST_Model(config, layers_details)
    data_gen = DataGenerator(config)
    iterator = data_gen.load('train')
    with tf.Session() as sess:
        mnist_trainer = MNISTTrainer(sess, model, data_gen, config,None)
