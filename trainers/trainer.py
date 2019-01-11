import sys
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

        self.iterator = self.data.load('train')
        self.sess.run(self.iterator.initializer)
        self.next_element = self.iterator.get_next()

        i = 0
        while i<50000//self.config['batch_size']:#400: #
            print(50000//self.config['batch_size'], "steps needs to be run")
            i += 1
            if i%100 ==0: print("----------On Step:", i, "---------")
            self.train_step()



    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        batch_data_point = self.sess.run(self.next_element)
        batch_images = batch_data_point[0]
        batch_labels = batch_data_point[1]

        batch_onehot_labels = self.sess.run(
            tf.one_hot(batch_labels,self.config["num_classes"]))

        # print("batch_labels", batch_labels)
        # print("batch_onehot_labels", batch_onehot_labels)

        loss, _optimization, accuracy = self.sess.run([self.model.loss, self.model.optimization, model.accuracy],
                                     feed_dict={
                                         self.model.X: batch_images,
                                         self.model.onehot_labels:
                                             batch_onehot_labels,
                                         self.model.learning_rate:
                                             self.config['learning_rate']
                                     })

        print("Loss:", loss, "Accuracy:", accuracy)

if __name__ == '__main__':

    model = MNIST_Model(config, layers_details)
    data_gen = DataGenerator(config)
    iterator = data_gen.load('train')
    with tf.Session() as sess:
        mnist_trainer = MNISTTrainer(sess, model, data_gen, config, None)
        epoch = 0
        while epoch < config['num_epoch']:
            mnist_trainer.train_epoch()
            epoch += 1
            print("Trained for", epoch, "epochs.")
    print("Training Completed!")


