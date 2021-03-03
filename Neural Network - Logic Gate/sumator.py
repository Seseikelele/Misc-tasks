# import numpy
from math import exp
from pprint import pprint
from random import random, sample, seed

### inspirowane https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

seed(1)
n_inputs = 3
n_hidden = 4
n_outputs = 2

network = [
	[{'weights': [random() for _ in range(n_inputs)]} for _ in range(n_hidden)],
	[{'weights': [random() for _ in range(n_hidden)]} for _ in range(n_outputs)]
]
print('BEFORE:')
for layer in network:
	pprint(layer)


def activate(weights, inputs):
	activation = 0
	for w, i in zip(weights, inputs):
		activation += w * i
	return activation


def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = list()
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs


def transfer_derivative(output):
	return output * (1.0 - output)


def backward_propagate_error(network, expected):
	network_size = len(network)
	for i in reversed(range(network_size)):
		layer = network[i]
		errors = list()
		if i != (len(network) - 1):
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += neuron['weights'][j] * neuron['delta']
				errors.append(error)
		else:
			for j, neuron in enumerate(layer):
				errors.append(expected[j] - neuron['output'])
		for j, neuron in enumerate(layer):
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
	for i, layer in enumerate(network):
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in layer:
			for j, input_ in enumerate(inputs):
				neuron['weights'][j] += l_rate * neuron['delta'] * input_
			neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
	train_indexes = [i for i in range(len(train))]
	epoch = 0
	sum_error = 9000
	# while epoch < n_epoch:
	while 0.28 < sum_error:
	# for epoch in range(n_epoch):
		sum_error = 0
		for i in sample(train_indexes, k=len(train)):
		# for row in train:
			row = train[i]
			outputs = forward_propagate(network, row['inputs'])
			expected = row['outputs']
			sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row['inputs'], l_rate)
		if epoch % 250 == 0:
			print(f'>epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}')
		epoch += 1


dataset = [
	{'inputs': [0, 0, 0], 'outputs': [0, 0]},
	{'inputs': [1, 0, 0], 'outputs': [1, 0]},
	{'inputs': [0, 1, 0], 'outputs': [1, 0]},
	{'inputs': [1, 1, 0], 'outputs': [0, 1]},
	{'inputs': [0, 0, 1], 'outputs': [1, 0]},
	{'inputs': [1, 0, 1], 'outputs': [0, 1]},
	{'inputs': [0, 1, 1], 'outputs': [0, 1]},
	{'inputs': [1, 1, 1], 'outputs': [1, 1]}
]

train_network(network, dataset, 0.133, 12000, n_outputs)
print('\nAFTER:')
for layer in network:
	pprint(layer)


def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs


def test(network, row):
	i1, i2, i3 = row['inputs']
	ex1, ex2 = row['outputs']
	o1, o2 = predict(network, [i1, i2, i3])
	print(f'inputs: {i1}, {i2}, {i3}, Expected: {ex1}, {ex2}, Got: {round(o1)}, {round(o2)} ({o1:.3f}, {o2:.3f})')


for row in dataset:
	test(network, row)
