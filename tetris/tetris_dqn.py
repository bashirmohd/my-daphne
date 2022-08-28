import time
import copy
import tensorflow as tf
import numpy as np
import random
from tetris import Env
from collections import deque
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

#twitter code
import tweepy
import pika


CONSUMER_KEY = "l12Sy11edD1SaLdQpQlCvBweM"
CONSUMER_SECRET = "sDwNUOw0CrZLCqpjB7fsJ5Tj62C7X1HpLvVqyETAalYPNafl6B"
ACCESS_TOKEN = "1116074345296617472-MOTvfQnYxohotK1ara05dQvU32MB3S"
ACCESS_TOKEN_SECRET = "OKWUn4Zz5482G4fujYdQQL4kbOD98ObFVLuG5V3llZEIq" 


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
#######

EPISODE = 50000
GAME_VELOCTY = 1#  0.000001
ACTION_VELOCITY = 1# 0.000001




ret = [[0] * 84 for _ in range(84)]

class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        #self.exploration_steps = 1000000.
        self.exploration_steps = 10000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end)/self.exploration_steps
        self.batch_size = 32
        self.train_start = 20000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # self.memory = deque(maxlen=20000)
        self.memory = []
        self.no_op_steps = 30
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.avg_q_max, self.avg_loss = 0, 0

    # Huber Loss
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        # lr = 0.00025
        optimizer = RMSprop(lr=0.001, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        #sequentila is a linear stack of layers

        #default activation: 'relu'
        #others: 'tanh', 'elu', selu', 'softmax'
        #.... scaled exponential linear unit
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        #batch size =32 and input shape = (84, 84, 4)

        '''Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None (glorot_uniform, he_normal), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        layer input and output of tensors
        use_bias =true,bais vector created and added
        filters =32 (dim of output filters in layer)
        kernel-size=(8,8) height and width of 2d convolution window
        strides=4,4= strides of convolution in height and width
        activation deafult = linear
        The kernel_regularizer parameter in particular is one that I adjust often to reduce overfitting and increase the ability for a model to generalize to unfamiliar images.
      
        from keras.regularizers import l2
        model.add(Conv2D(32, (3, 3), activation="relu"),
        kernel_regularizer=l2(0.0005))
        The amount of regularization you apply is a hyperparameter you will need to tune for your own dataset, but I find values of 0.0001-0.001 are good ranges to start with.

        I would suggest leaving your bias regularizer alone — regularizing the bias typically has very little impact on reducing overfitting.



        '''

        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        # #added
        # model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # <s, a, r, s'>
    def append_sample(self, history, action, reward, next_history):
        self.memory.append((history, action, reward, next_history))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward = [], []
       
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
        channel.queue_declare(queue='loss')
        channel.basic_publish(exchange='',routing_key='loss',body=loss)
        print(" [x] Sent 'Hello World!'")

        connection.close()

        #print("train model")
        #print(history, action, reward, loss)



    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        tf.train.AdamOptimizer
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


def pre_processing(curr_map,curr_block_pos):
    copy_map = copy.deepcopy(curr_map)
    ny, nx = 4.20, 10.5
    for n in curr_block_pos:
        copy_map[n[0]][n[1]] = 1
    for n in range(20):
        for m in range(8):
            for i in range(int(n * ny), int(n * ny + ny)):
                for j in range(int(m * nx), int(m * nx + nx)):
                    ret[i][j] = copy_map[n][m]
    return ret


if __name__ == "__main__":
    tetris = Env()
    agent = DQNAgent(action_size=3)


    auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    #tweet ="Test 123"
    #status=api.update_status(status=tweet)

    state = pre_processing(tetris.map, tetris._get_curr_block_pos())
    #print(state)
    history = np.stack((state, state, state, state), axis = 2)
    history = np.reshape([history], (1, 84, 84, 4))
    #print(history)

    #initialize state and history

    start_time = time.time()
    action_time = time.time()
    global_step = 0

    for epi in range(EPISODE):
        step = 0
        while True:
            end_time = time.time()

            if end_time - action_time >= ACTION_VELOCITY:
                global_step += 1
                step += 1
                action = agent.get_action(history)
                reward = tetris.step(action)

                next_state = pre_processing(tetris.map, tetris._get_curr_block_pos())
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])
                agent.append_sample(history, action, reward, next_history)
                print("----------")
                print ("step number ", step)
                print("3 action:", action)
                print("4 reward:" , reward)

               # print("history: ", history)
               # print("new values")
               # print(history, action, reward, next_history)

                if len(agent.memory) >= agent.batch_size:
                    print("length agent memory: ", len(agent.memory))
                    print("length batch size: ", agent.batch_size)
                    agent.train_model()

                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

                history = next_history

                action_time = time.time()

            if end_time - start_time >= GAME_VELOCTY:
                # game over
                if tetris.is_game_end():
                    '''
                    if global_step > agent.train_start:
                        stats = [tetris.score, agent.avg_q_max / float(global_step), global_step,
                                 agent.avg_loss / float(global_step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, epi + 1)
                    '''
                    print('episode:{}, score:{}, epsilon:{}, global step:{}, avg_qmax:{}, memory:{}'.
                          format(epi, tetris.score, agent.epsilon, global_step,
                                 agent.avg_q_max / float(step), len(agent.memory)))
                    tetris.reset()
                    agent.avg_q_max, agent.avg_loss = 0, 0
                    break
                else:
                    buffer = tetris.step(0)
                start_time = time.time()