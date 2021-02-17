import numpy as np
import tensorflow as tf

#incomplete

class DeepQNetworkAgent:
	def __init__(
		self, 
		n_actions, 
		n_features, 
		learning_rate=0.01, 
		reward_decay=0.9, 
		e_greedy=0.9,
		replace_target_iter=300,
		memory_size=500,
		batch_size=32,
		e_greedy_increment=None,
		output_graph=False
	):

		self.n_actions = n_actions #sell buy hold
		self.n_features = n_features #previous days
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		#total learning step
		self.learning_step_counter=0

		#initialize memory to zero [state,action,reward,state_]
		self.memory = np.zeros((self.memory_size, n_features * 2 +2))

		#define the main network
		self.main_network = self._build_network()

		#define the target network
		self.target_network = self._build_network()

		#copy the weights of the main network to the target network
		self.target_network.set_weights(self.main_network.get_weights())


	#Let's define a function called build_network which is essentially our DQN. 
	def build_network(self):
		model = Sequential()
		model.add(Dense(units=128, activation="relu", input_dim=self.n_features))
		model.add(Dense(units=256, activation="relu"))
		model.add(Dense(units=256, activation="relu"))
		model.add(Dense(units=128, activation="relu"))
		model.add(Dense(units=self.action_size))

		model.compile(loss=self.loss, optimizer=self.optimizer)
		return model