import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # Amount of pictures loaded at once

# Placeholder for data being added to network
# Height by width
x = tf.placeholder('float',[None, 784]) # Data
										# 28x28 pixels as individuals
										# Will throw error if wrong size file is included
y = tf.placeholder('float',[None, 784]) # Data label

def neural_network_model(data): # Raw input data
	
	# (inpt_data * weights) + biases

	hidden_1_layer = {	'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
						'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}
