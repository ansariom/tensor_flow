'''
Created on Jun 9, 2017

@author: mitra
'''
import tensorflow as tf


def add_example():
    print("Example1 -----")
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)

    print (node1, node2)
    
    sess = tf.Session()
    print(sess.run([node1, node2]))
    
def add_example2():
    print("Example2------")
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b # + provides a shortcut for tf.add(a, b)
    
    # Use dict to feed in values to these two placeholders
    sess = tf.Session()
    print(sess.run(adder_node, {a:3, b:4.5}))
    print(sess.run(adder_node, {a:[1,3], b:[4.5, 2]}))
    
def add_example3():
    print("Example3------")
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b # + provides a shortcut for tf.add(a, b)
    adder_triple = adder_node * 3
    
    # Use dict to feed in values to these two placeholders
    sess = tf.Session()
    print(sess.run(adder_triple, {a:3, b:4.5}))
    print(sess.run(adder_triple, {a:[1,3], b:[4.5, 2]}))
    
def learn_example1():
    print("Learning Example1")
    import numpy as np
    import tensorflow as tf
    
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]
    # training loop
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
      sess.run(train, {x:x_train, y:y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    

if __name__ == "__main__":
    add_example()
    add_example2()
    add_example3()
    learn_example1()