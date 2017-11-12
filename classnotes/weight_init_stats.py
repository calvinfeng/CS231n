import numpy as np
import matplotlib.pyplot as plt

# Assume some unit Gaussian 10-D input data

D = np.random.randn(1000, 500) # Randomly initialized inputs
hidden_layer_sizes = [500]*10 # 10 hidden layers

nonlinearities = ['tanh']*len(hidden_layer_sizes)

act_func = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

Hs = dict()

def small_init(in_dimen, out_dimen):
    return np.random.randn(in_dimen, out_dimen) * 0.01

def big_init(in_dimen, out_dimen):
    return np.random.randn(in_dimen, out_dimen) * 1

def xavier_init(in_dimen, out_dimen):
    return np.random.randn(in_dimen, out_dimen) / np.sqrt(in_dimen)

for i in range(len(hidden_layer_sizes)):
    # Input is D
    if i == 0:
        X = D
    else:
        X = Hs[i - 1]

    fan_in = X.shape[1]
    print("fan_in: %s" % fan_in)

    fan_out = hidden_layer_sizes[i]
    print("fan_out: %s" % fan_out)

    W = xavier_init(fan_in, fan_out) # Layer weights initialization

    H = np.dot(X, W) # Matrix multiplication
    H = act_func[nonlinearities[i]](H)
    Hs[i] = H

print('Input layer had mean %f and std %f' % (np.mean(D), np.std(D)))

layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]

for i, H in Hs.items():
    print('Hidden layer %d had mean %f and std %f' % (i + 1, layer_means[i], layer_stds[i]))

# Plot the results
plt.figure()
plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')

# Plot the raw distribution
plt.figure()
for i, H in Hs.items():
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(H.ravel(), 30, range=(-1, 1))

plt.show()

"""
As we can see that, if we initialize the weight with small random numbers using Gaussian distribution, as it approaches
deep into the layers, all activiations go toward zero!!!
"""
