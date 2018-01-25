import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Define the constants
mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128

# Read the input data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True, seed=42, validation_size=0)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev, seed=42)

# Set up the discriminator variables
X = tf.placeholder(tf.float32, shape=[None, X_dim], name="x")

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]), name="dw1")
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="db1")

D_W2 = tf.Variable(xavier_init([h_dim, 1]), name="dw2")
D_b2 = tf.Variable(tf.zeros(shape=[1]), name="db2")

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Set up the generator variables
z = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]), name="gw1")
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]), name="gb1")

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]), name="gw2")
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]), name="gb2")

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_z(m, n):
    np.random.seed(42)
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


# Set up the actual model
G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

# Set up the loss functions
D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = -tf.reduce_mean(D_fake)

# Set up the optimizer
D_opt = tf.train.RMSPropOptimizer(learning_rate=1E-4)
G_opt = tf.train.RMSPropOptimizer(learning_rate=1E-4)

D_solver = D_opt.minimize(D_loss, var_list=theta_D)
G_solver = G_opt.minimize(G_loss, var_list=theta_G)
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Save the initial values of each variable
np.savetxt('dw1.txt', D_W1.eval(sess))
np.savetxt('dw2.txt', D_W2.eval(sess))
np.savetxt('gw1.txt', G_W1.eval(sess))
np.savetxt('gw2.txt', G_W2.eval(sess))

if not os.path.exists('out/'):
    os.makedirs('out/')

# Save the initial batch
X_mb, _ = mnist.train.next_batch(mb_size)
np.savetxt('xmb.txt', X_mb)

# Save the first set of samples
testZ = sample_z(mb_size, z_dim)
np.savetxt('testZ.txt', testZ)

# Save the gradients of each variable
grads_and_vars = D_opt.compute_gradients(D_loss, var_list=theta_D)
for gv in grads_and_vars:
    grad = sess.run(gv[0], feed_dict={X: X_mb, z: testZ})
    var  = gv[1].name[:-2]
    np.savetxt('grad_' + var + '.txt', grad)
grads_and_vars = G_opt.compute_gradients(G_loss, var_list=theta_G)
for gv in grads_and_vars:
    grad = sess.run(gv[0], feed_dict={X: X_mb, z: testZ})
    var  = gv[1].name[:-2]
    np.savetxt('grad_' + var + '.txt', grad)

# Update the discriminator
sess.run([D_solver, clip_D], feed_dict={X: X_mb, z: testZ})

# Save the updated values of each discriminator variable
np.savetxt('dw12.txt', D_W1.eval(sess))
np.savetxt('db12.txt', D_b1.eval(sess))
np.savetxt('dw22.txt', D_W2.eval(sess))
np.savetxt('db22.txt', D_b2.eval(sess))

# Pull a new sample
testZ2 = sample_z(mb_size, z_dim)
np.savetxt('testZ2.txt', testZ2)

# Update the generator
sess.run([G_solver], feed_dict={z: testZ2})

# Save the update values of each generator variable
np.savetxt('gw12.txt', G_W1.eval(sess))
np.savetxt('gb12.txt', G_b1.eval(sess))
np.savetxt('gw22.txt', G_W2.eval(sess))
np.savetxt('gb22.txt', G_b2.eval(sess))

# Calculate the new loss values
D_loss_curr, G_loss_curr = sess.run(
    [D_loss, G_loss],
    feed_dict={X: X_mb, z:testZ2}
)

print('Iter: {}; G_loss: {:.8}; D loss: {:.8}'
    .format(0, G_loss_curr, D_loss_curr))
