import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from problems import create_params_as, assign_params, vector_gradients, var_shape
import pdb

def sgd_log(num_updates, track_every):
    learning_rate = 0.005
    training_epochs = 25
    gamma = 1e-4

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.nn.softmax(tf.matmul(x, W) + b)

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) + gamma * tf.reduce_sum(tf.square(W))
    [dW, db] = tf.gradients(cost, [W, b])


def svrg_log(num_updates, track_every, m=100, learning_rate=0.025):
    ### Hyperparameters
    gamma = 1e-4  # Strength of L2 regularization
    num_iterations = 10  # For total number of iterations, multiply by m

    ### tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

    ### Placeholder dictating whether SVRG or SGD update should be applied
    svrg_flag = tf.placeholder(tf.bool)

    ### Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    ### Variables to be used for our control variate
    control_W = tf.Variable(tf.zeros([784, 10]))
    control_b = tf.Variable(tf.zeros([10]))
    batch_dcontrol_W = tf.Variable(tf.zeros([784, 10]))
    batch_dcontrol_b = tf.Variable(tf.zeros([10]))

    ### Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
    control_pred = tf.nn.softmax(tf.matmul(x, control_W) + control_b)

    ### Define loss function
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) + gamma * tf.reduce_sum(
        tf.square(W))  # Cost for actual weights
    control_cost = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(control_pred), reduction_indices=1)) + gamma * tf.reduce_sum(
        tf.square(control_W))  # Cost for control weights

    ### Get gradients
    [dW, db] = tf.gradients(cost, [W, b])
    [dcontrol_W, dcontrol_b] = tf.gradients(control_cost, [control_W, control_b])

    ### Form SVRG and SGD updates
    update_W = tf.cond(svrg_flag, lambda: learning_rate * (dW - dcontrol_W + batch_dcontrol_W),
                       lambda: learning_rate * dW)
    update_b = tf.cond(svrg_flag, lambda: learning_rate * (db - dcontrol_b + batch_dcontrol_b),
                       lambda: learning_rate * db)

    ### Adjust parameters based on computed update
    adj_W = W.assign(W - update_W)
    adj_b = b.assign(b - update_b)

    ### Create nodes for updating our control variate (done every m iterations)
    reset_W = control_W.assign(W)
    reset_b = control_b.assign(b)
    reset_batch_db = batch_dcontrol_b.assign(db)
    reset_batch_dW = batch_dcontrol_W.assign(dW)


    ### Train our model
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        ### We run one iteration of SGD to initialize the weights for SVRG
        idx = np.random.randint(low=0, high=n)
        curr_cost = sess.run(cost, feed_dict = {x:trX, y:trY})
        print(curr_cost)
        sess.run([adj_W, adj_b], feed_dict={x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: False})
        sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW], feed_dict={x: trX, y: trY})

        costs = []
        variances = []
        i = 0
        curr_cost = sess.run(cost, feed_dict = {x:trX, y:trY})
        print(curr_cost)
        while (i < num_updates):
            t = 0
            while (t < m and i < num_updates):
                ### Apply control variate update
                idx = np.random.randint(low=0, high=n)
                sess.run([adj_b, adj_W], feed_dict={x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: True})

                ###Track cost/variance when specified
                if (i % track_every == 0):
                    curr_cost, gt_update = sess.run([cost, update_W], feed_dict={x: trX, y: trY, svrg_flag: True})
                    costs.append(curr_cost)

                    ### Estimate variance of our update
                    curr_variance = 0
                    for j in range(n):
                        # idx = np.random.randint(low = 0, high = n)
                        curr_update = update_W.eval(feed_dict={x: trX[j, None, :], y: trY[j, None, :], svrg_flag: True})
                        curr_variance += np.sum(np.square(curr_update - gt_update))
                    variances.append(curr_variance)
                    #print(curr_variance)
                    print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))

                ### Increment our counters
                t += 1
                i += 1

            ### Update our control values every m iterations
            sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW], feed_dict={x: trX, y: trY})

    return costs, variances


def sarah_log(num_updates, track_every, m=100, learning_rate = 0.5):
    ### Hyperparameters
    cur_time=time.time()
    m = m  # How many iterations until recalculate batch gradient

    gamma = 1e-4  # Strength of L2 regularization
    num_iterations = 10  # For total number of iterations, multiply by m

    ### tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

    ### Placeholder dictating whether SVRG or SGD update should be applied
    svrg_flag = tf.placeholder(tf.bool)

    ### Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    ### Variables to be used for our control variate
    control_W = tf.Variable(tf.zeros([784, 10]))
    control_b = tf.Variable(tf.zeros([10]))
    batch_dcontrol_W = tf.Variable(tf.zeros([784, 10]))
    batch_dcontrol_b = tf.Variable(tf.zeros([10]))
    mid_W = tf.Variable(tf.zeros([784, 10]))
    mid_b = tf.Variable(tf.zeros([10]))

    ### Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
    control_pred = tf.nn.softmax(tf.matmul(x, control_W) + control_b)

    ### Define loss funcFtion
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) + gamma * tf.reduce_sum(
        tf.square(W))  # Cost for actual weights
    control_cost = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(control_pred), reduction_indices=1)) + gamma * tf.reduce_sum(
        tf.square(control_W))  # Cost for control weights

    ### Get gradients
    [dW, db] = tf.gradients(cost, [W, b])
    [dcontrol_W, dcontrol_b] = tf.gradients(control_cost, [control_W, control_b])

    ### Form SVRG and SGD updates
    update_W = tf.cond(svrg_flag, lambda: (dW - dcontrol_W + batch_dcontrol_W),
                       lambda:  dW)
    update_b = tf.cond(svrg_flag, lambda: (db - dcontrol_b + batch_dcontrol_b),
                       lambda: db)

    ### Adjust parameters based on computed update
    adj_W = W.assign(W - learning_rate * update_W)
    adj_b = b.assign(b - learning_rate * update_b)
    save_W = mid_W.assign(W)
    save_b = mid_b.assign(b)
    ret_W = control_W.assign(mid_W)
    ret_b = control_b.assign(mid_b)

    ### Create nodes for updating our control variate (done every m iterations)
    reset_W = control_W.assign(W)
    reset_b = control_b.assign(b)
    reset_batch_db = batch_dcontrol_b.assign(db)
    reset_batch_dW = batch_dcontrol_W.assign(dW)
    reset_batch_dcontrol_W = batch_dcontrol_W.assign(update_W)
    reset_batch_dcontrol_b = batch_dcontrol_b.assign(update_b)

    ### Train our model
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        curr_cost = sess.run(cost, feed_dict = {x:trX, y:trY})
        print(curr_cost)
        ### We run one iteration of SGD to initialize the weights for SVRG
        idx = np.random.randint(low=0, high=n)
        sess.run([adj_W, adj_b], feed_dict={x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: False})
        curr_cost = sess.run(cost, feed_dict = {x:trX, y:trY})
        print(curr_cost)
        sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW, adj_W, adj_b], feed_dict={x: trX, y: trY, svrg_flag:False})
        costs = []
        vars = []
        vs = []
        i = 0
        curr_cost = sess.run(cost, feed_dict = {x:trX, y:trY})
        print(curr_cost)
        while (i < num_updates):
            t = 0
            while (t < m and i < num_updates):
                #pdb.set_trace()
                ### Apply control variate update
                idx = np.random.randint(low=0, high=n)
                #uu, vv, o = sess.run([b, control_b, batch_dcontrol_b])
                sess.run([save_W, save_b])
                #uu, vv, o = sess.run([b, control_b, batch_dcontrol_b])
                sess.run([adj_b, adj_W], feed_dict={x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: True})
                curr_cost, update_v = sess.run([cost, update_W], feed_dict={x: trX, y: trY, svrg_flag: True})
                #uu, vv, o = sess.run([b, control_b, batch_dcontrol_b])
                if (i % track_every == 0):
                    curr_variance = 0
                    for j in range(n):
                        ge_v = sess.run(update_W, feed_dict = {x:trX[idx, None, :], y:trY[idx, None, :], svrg_flag:True})
                        curr_variance += np.sum(np.square(ge_v-update_v))
                    costs.append(curr_cost)
                    vars.append(curr_variance)
                    # print(i)
                    print(curr_cost)
                    print(curr_variance)
                sess.run([reset_batch_dcontrol_W, reset_batch_dcontrol_b], feed_dict={x: trX[idx, None, :], y: trY[idx, None, :], svrg_flag: True})
                #uu, vv, o = sess.run([b, control_b, batch_dcontrol_b])
                sess.run([ret_W, ret_b])
                #uu, vv, o = sess.run([b, control_b, batch_dcontrol_b])

                ###Track cost/variance when specified
                """
                if (i % track_every == 0):

                    curr_cost, gt_update = sess.run([cost, update_W], feed_dict={x: trX, y: trY, svrg_flag: True})
                    costs.append(curr_cost)

                    ### Estimate variance of our update
                    curr_variance = 0
                    for j in range(n):
                        # idx = np.random.randint(low = 0, high = n)
                        curr_update = update_W.eval(feed_dict={x: trX[j, None, :], y: trY[j, None, :], svrg_flag: True})
                        #print(curr_update)
                        curr_variance += np.sum(np.square(curr_update - gt_update))
                    variances.append(curr_variance)
                """

                    #print("Iteration: {0}\tCost: {1}\tVariance: {2}".format(i, curr_cost, curr_variance))

                ### Increment our counters
                t += 1
                i += 1
            #i += 1

            ### Update our control values every m iterations
            sess.run([reset_W, reset_b, reset_batch_db, reset_batch_dW], feed_dict={x: trX, y: trY})

    return costs, vars

class sarah_optimizer(object):
    def __init__(self, learning_rate, params, loss, control_params, control_loss):
        self.learning_rate = learning_rate
        self.params = params
        self.control_params = control_params
        self.current_gradients = create_params_as(params, name='current_gradients')
        self.mid_params = create_params_as(params, name='mid_params')
        self.loss = loss
        self.control_loss = control_loss
        self.gradients = tf.gradients(self.loss, self.params)
        #pdb.set_trace()
        self.control_gradients = tf.gradients(self.control_loss, self.control_params)
        self.update = [tf.add(tf.subtract(a,b), c) for a,b,c in zip(self.gradients, self.control_gradients, self.current_gradients)]
        #self.update = [tf.subtract(a, tf.add(b, c)) for a,b,c in zip(self.gradients, self.control_gradients, self.current_gradients)]
        #self.update = tf.subtract(self.gradients,tf.add(self.control_gradients,self.current_gradients))


    def minimize(self):
        opt_1 = assign_params(self.current_gradients, self.update)
        with tf.control_dependencies(opt_1):
            opt_2 = assign_params(self.mid_params, self.params)
            with tf.control_dependencies(opt_2):
                opt_3 = assign_params(self.params, [tf.subtract(a, tf.multiply(self.learning_rate, b)) for a,b in zip(self.params, self.update)])
                #opt_3 = assign_params(self.params, tf.subtract(self.params,tf.multiply(self.learning_rate,self.update)))
                with tf.control_dependencies(opt_3):
                    opt_4 = assign_params(self.control_params, self.mid_params)
        return opt_4
    def upd(self):
        opt_1 = assign_params(self.control_params, self.params)
        with tf.control_dependencies(opt_1):
            opt_2 = assign_params(self.current_gradients, self.gradients)
            with tf.control_dependencies(opt_2):
                opt_3 = assign_params(self.params, [tf.subtract(a, tf.multiply(self.learning_rate, b)) for a,b in zip(self.params, self.update)])

        return opt_3

class svrg_optimizer(object):
    def __init__(self, learning_rate, params, loss, control_params, control_loss):
        self.learning_rate = learning_rate
        self.params = params
        self.control_params = control_params
        self.current_gradients = create_params_as(params, name='current_gradients')
        self.mid_params = create_params_as(params, name='mid_params')
        self.loss = loss
        self.control_loss = control_loss
        self.gradients = tf.gradients(self.loss, self.params)
        #pdb.set_trace()
        self.control_gradients = tf.gradients(self.control_loss, self.control_params)
        self.update = [tf.add(tf.subtract(a,b), c) for a,b,c in zip(self.gradients, self.control_gradients, self.current_gradients)]
        #self.update = [tf.subtract(a, tf.add(b, c)) for a,b,c in zip(self.gradients, self.control_gradients, self.current_gradients)]
        #self.update = tf.subtract(self.gradients,tf.add(self.control_gradients,self.current_gradients))


    def minimize(self):
        opt_3 = assign_params(self.params, [tf.subtract(a, tf.multiply(self.learning_rate, b)) for a,b in zip(self.params, self.update)])

        return opt_3
    def upd(self):
        opt_1 = assign_params(self.control_params, self.params)
        with tf.control_dependencies(opt_1):
            opt_2 = assign_params(self.current_gradients, self.gradients)
        return opt_2
class scgd_optimizer(object):
    def __init__(self, alpha, model, beta=0.9, name='scgd'):
        with tf.variable_scope(name) as scope:
            self.beta = beta
            self.learning_rate = alpha
            self.params = model.params

            self.g = model.out_g
            self.grad_g = vector_gradients(self.g, self.params)

            self.inp_g = model.inp_g
            #self.current_g = create_params_as(self.g, name='current_g')
            self.grad_f = tf.gradients(model.out_f, self.inp_g)[0]

            #self.update_g = [tf.add(tf.multiply((1-beta), a), tf.multiply(beta, b)) for a,b in zip(self.inp_g, self.g)]
            self.update_g = (1-beta)* self.inp_g + beta * self.g
                #(1-beta) * self.inp_g + beta * self.g
            self.update_f = alpha * tf.matmul(tf.transpose(self.grad_g) , self.grad_f)

    def update(self):
        opt1 = assign_params(self.inp_g, self.update_g)
        with tf.control_dependencies([opt1]):
            flat_params = tf.concat(values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(self.params, self.params)], axis=0)
            opt2 = assign_params(self.params, flat_params - self.learning_rate * tf.reshape(self.update_f, flat_params.get_shape()))

            #opt2 = assign_params(self.params, [tf.subtract(a,b) for a,b in zip(self.params - self.update_f)])
        return opt2
class vrscpg_optimizer(object):
    def __init__(self, learning_rate, model, control_model):
        with tf.variable_scope('svrg') as scope:
            self.learning_rate = learning_rate
            self.params = model.params
            self.control_params = control_model.params

            self.g = model.out_g
            #pdb.set_trace()
            self.grad_g  = vector_gradients(self.g, self.params)
            self.grad_g = tf.reshape(self.grad_g, [self.grad_g.get_shape()[0], self.grad_g.get_shape()[1]])

            self.control_g = control_model.out_g
            self.control_grad_g = vector_gradients(self.control_g, self.control_params)
            self.control_grad_g = tf.reshape(self.control_grad_g, [self.control_grad_g.get_shape()[0], self.control_grad_g.get_shape()[1]])

            self.inp_g = model.inp_g
            self.control_inp_g = control_model.inp_g
            #pdb.set_trace()
            self.grad_f = tf.gradients(model.out_f, self.inp_g)[0]
            self.control_grad_f = tf.gradients(control_model.out_f, self.control_inp_g)[0]

            self.current_g = create_params_as(self.g, name='current_g')[0]
            self.current_grad_g = create_params_as(self.grad_g, name='current_grad_g')[0]
            self.current_grad_f = create_params_as(self.grad_f, name='current_grad_f')[0]

            self.g_estim = self.g - self.control_g + self.current_g
            self.grad_g_estim = self.grad_g - self.control_grad_g + self.current_grad_g
            self.grad_f_estim = self.grad_f - self.control_grad_f + self.current_grad_f
            #pdb.set_trace()

            self.f_gradients = tf.matmul(tf.transpose(self.grad_g), self.grad_f)
            self.current_f_gradients = create_params_as(self.f_gradients, name='current_f_gradients')[0]
            self.param_updates = tf.matmul(tf.transpose(self.grad_g_estim),self.grad_f) - tf.matmul(tf.transpose(self.current_grad_g), self.control_grad_f) + self.current_f_gradients
            #self.g_estim = [tf.add(tf.subtract(a, b), c) for a,b,c in zip([self.g], [self.control_g], [self.current_g])]

            #self.grad_g_estim = [tf.add(tf.subtract(a,b),c) for a,b,c in zip(self.grad_g, self.control_grad_g, self.current_grad_g)]
            #self.grad_f_estim = [tf.add(tf.subtract(a,b),c) for a,b,c in zip(self.grad_f, self.control_grad_f, self.current_grad_f)]

            #self.param_updates = [tf.subtract(tf.add(tf.matmul(a, b), tf.matmul(c, d)), e) for a,b,c,d,e in zip(self.grad_g_estim, self.grad_f, self.current_grad_g, self.control_grad_f, self.current_grad_f)]
                #self.grad_g_estim * self.grad_f + self.current_grad_g + self.control_grad_f - self.current_grad_f
    def minimize(self):
        ### update the parameters, update the estimate of g
        opt1 = assign_params(self.inp_g, self.g_estim)
        with tf.control_dependencies([opt1]):
            #pdb.set_trace()
            flat_params = tf.concat(values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(self.params, self.params)], axis=0)
            opt2 = assign_params(self.params, flat_params - self.learning_rate * tf.reshape(self.param_updates, flat_params.get_shape()))

        return opt2
    def upd(self):
        ### Two things to update: update the current fixed point, update the control parameters
        opt1 = assign_params(self.current_g, self.g)
        #pdb.set_trace()n([optimizer.g_estim, optimi
        with tf.control_dependencies([opt1]):
            opt2 = assign_params(self.control_inp_g, self.g)
            with tf.control_dependencies([opt2]):
                opt3 = assign_params(self.inp_g, self.g)
                with tf.control_dependencies([opt3]):
                    opt4 = assign_params(self.control_params, self.params)
                    with tf.control_dependencies(opt4):
                        opt5 = assign_params(self.current_grad_f, self.grad_f)
                        with tf.control_dependencies([opt5]):
                            opt6 = assign_params(self.current_grad_g, self.grad_g)
                            with tf.control_dependencies([opt6]):
                                opt7 = assign_params(self.current_f_gradients, self.f_gradients)

        return [opt7]

class spider_scgd_optimizer(object):
    def __init__(self, learning_rate, model, control_model):
        with tf.variable_scope('spider') as scope:
            self.learning_rate = learning_rate
            self.params = model.params
            self.control_params = control_model.params
            self.mid_params = create_params_as(self.params, name='mid_params')

            self.g = model.out_g
            self.epochs = 0
            # pdb.set_trace()
            self.grad_g = vector_gradients(self.g, self.params)
            self.grad_g = tf.reshape(self.grad_g, [self.grad_g.get_shape()[0], self.grad_g.get_shape()[1]])

            self.control_g = control_model.out_g
            self.control_grad_g = vector_gradients(self.control_g, self.control_params)
            self.control_grad_g = tf.reshape(self.control_grad_g,
                                             [self.control_grad_g.get_shape()[0], self.control_grad_g.get_shape()[1]])

            self.inp_g = model.inp_g
            self.control_inp_g = control_model.inp_g
            # pdb.set_trace()
            self.grad_f = tf.gradients(model.out_f, self.inp_g)[0]
            self.control_grad_f = tf.gradients(control_model.out_f, self.control_inp_g)[0]

            self.current_g = create_params_as(self.g, name='current_g')[0]
            self.current_grad_g = create_params_as(self.grad_g, name='current_grad_g')[0]
            self.current_grad_f = create_params_as(self.grad_f, name='current_grad_f')[0]

            self.g_estim = self.g - self.control_g + self.current_g
            self.grad_g_estim = self.grad_g - self.control_grad_g + self.current_grad_g
            self.grad_f_estim = self.grad_f - self.control_grad_f + self.current_grad_f
            # pdb.set_trace()


            self.f_gradients = tf.matmul(tf.transpose(self.grad_g), self.grad_f)
            self.current_f_gradients = create_params_as(self.f_gradients, name='current_f_gradients')[0]

            self.mid_g_estim = create_params_as(self.g_estim, name='mid_g_estim')[0]
            self.mid_grad_g_estim = create_params_as(self.grad_g_estim, name='grad_g_estim')[0]
            self.mid_grad_f_estim = create_params_as(self.grad_f_estim, name='grad_f_estim')[0]
            self.mid_f_gradients = create_params_as(self.f_gradients, name='mid_f_gradients')[0]

            self.param_updates = tf.matmul(tf.transpose(self.grad_g_estim), self.grad_f) - tf.matmul(
                tf.transpose(self.current_grad_g), self.control_grad_f) + self.current_f_gradients
            self.param_updates1 = tf.matmul(tf.transpose(self.grad_g_estim), self.grad_f)
    #self.grad_g_estim * self.grad_f + self.current_grad_g + self.control_grad_f - self.current_grad_f
    def minimize(self):
        ### Two extra operations are needed: update currents, update control
        opt1 = assign_params(self.inp_g, self.g_estim)
        with tf.control_dependencies([opt1]):
            opt2 = assign_params(self.mid_params, self.params)
            with tf.control_dependencies(opt2):
                opt3 = assign_params(self.mid_g_estim, self.g_estim)
                opt4 = assign_params(self.mid_grad_g_estim, self.grad_g_estim)
                opt5 = assign_params(self.mid_grad_f_estim, self.grad_f_estim)
                opt6 = assign_params(self.mid_f_gradients, self.param_updates)
                #pdb.set_trace()
                with tf.control_dependencies([opt3, opt4, opt5, opt6]):
                    flat_params = tf.concat(
                        values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(self.params, self.params)],
                        axis=0)
                    tmp = tf.reshape(self.param_updates, flat_params.get_shape())
                    opt7 = assign_params(self.params, flat_params - self.learning_rate * 20 * np.power(0.5, self.epochs)/tf.norm(tmp, ord=2) * tmp)
                    with tf.control_dependencies(opt7):
                        opt8 = assign_params(self.current_g, self.mid_g_estim)
                        with tf.control_dependencies([opt8]):
                            opt9 = assign_params(self.current_grad_g, self.mid_grad_g_estim)
                            with tf.control_dependencies([opt9]):
                                opt10 = assign_params(self.current_grad_f, self.mid_grad_f_estim)
                                with tf.control_dependencies([opt10]):
                                    opt11 = assign_params(self.current_f_gradients, self.mid_f_gradients)
                                    with tf.control_dependencies([opt11]):
                                        opt12 = assign_params(self.control_params, self.mid_params)
        return opt12
    def minimize1(self):
        ### Two extra operations are needed: update currents, update control
        opt1 = assign_params(self.inp_g, self.g_estim)
        with tf.control_dependencies([opt1]):
            opt2 = assign_params(self.mid_params, self.params)
            with tf.control_dependencies(opt2):
                opt3 = assign_params(self.mid_g_estim, self.g_estim)
                opt4 = assign_params(self.mid_grad_g_estim, self.grad_g_estim)
                opt5 = assign_params(self.mid_grad_f_estim, self.grad_f_estim)
                opt6 = assign_params(self.mid_f_gradients, self.param_updates1)
                with tf.control_dependencies([opt3, opt4, opt5, opt6]):
                    flat_params = tf.concat(
                        values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(self.params, self.params)],
                        axis=0)
                    tmp = tf.reshape(self.param_updates1, flat_params.get_shape())
                    opt7 = assign_params(self.params, flat_params - self.learning_rate * 20 * np.power(0.5, self.epochs)/tf.norm(tmp, ord=2) * tmp)
                    with tf.control_dependencies(opt7):
                        opt8 = assign_params(self.current_g, self.mid_g_estim)
                        with tf.control_dependencies([opt8]):
                            opt9 = assign_params(self.current_grad_g, self.mid_grad_g_estim)
                            with tf.control_dependencies([opt9]):
                                opt10 = assign_params(self.current_grad_f, self.mid_grad_f_estim)
                                with tf.control_dependencies([opt10]):
                                    opt11 = assign_params(self.current_f_gradients, self.mid_f_gradients)
                                    with tf.control_dependencies([opt11]):
                                        opt12 = assign_params(self.control_params, self.mid_params)
        return opt12
    def upd(self):
        #pdb.set_trace()
        self.epochs = self.epochs + 1
        ### Two things to update: update the current fixed point, update the control parameters
        opt1 = assign_params(self.current_g, self.g)
        #pdb.set_trace()n([optimizer.g_estim, optimi
        with tf.control_dependencies([opt1]):
            opt2 = assign_params(self.control_inp_g, self.g)
            with tf.control_dependencies([opt2]):
                opt3 = assign_params(self.inp_g, self.g)
                with tf.control_dependencies([opt3]):
                    opt4 = assign_params(self.control_params, self.params)
                    with tf.control_dependencies(opt4):
                        opt5 = assign_params(self.current_grad_f, self.grad_f)
                        with tf.control_dependencies([opt5]):
                            opt6 = assign_params(self.current_grad_g, self.grad_g)
                            with tf.control_dependencies([opt6]):
                                opt7 = assign_params(self.current_f_gradients, self.f_gradients)
        return [opt7]