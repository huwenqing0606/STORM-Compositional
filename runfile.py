from sgd import scgd_optimizer, vrscpg_optimizer, spider_scgd_optimizer
from problems import create_mlp_log, forward_mlp_log, create_params_as
import tensorflow as tf
import numpy as np
import pdb
import pickle
import time
import math
from tensorflow.examples.tutorials.mnist import input_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--layers", type=int, help="the number of mlp layers", default=1)
parser.add_argument("-hs", "--hidden_size", type=int, help="300 or 100", default=50)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=0.005)
parser.add_argument("-e", "--epochs", type=float, help='num_epochs', default=500)
parser.add_argument("-m", "--minibatchsize", help="mini batch size", default=5)
parser.add_argument("-t", "--task", help="mnist or cifar10", default="rl")
parser.add_argument("-p", "--plus", help="is it sarah+ or not", default=0)
parser.add_argument('-act', '--activation', default=tf.identity)
parser.add_argument('-bs','--batch_size', default=20)

args = parser.parse_args()

Filename = 'spider_tests' + args.task + '_'+ str(args.layers) + str(time.time())
#pdb.set_trace()
if int(args.plus) == True:
    Filename = '+' + Filename
Filename += '.txt'
print(Filename)

trX = np.random.random([400, 200])
trX = np.concatenate([trX, np.ones([400, 1])], axis=1)
trX = trX.astype(np.float32)


s2a = 5
s2b = 1


n = trX.shape[0]

policy = np.random.random([400, 10])
for i in range(400):
    policy[i] = policy[i]/np.sum(policy[i])
policy = policy.astype(np.float32)

tran = np.random.random([400, 10, 400])
for i in range(400):
    for j in range(10):
        tran[i][j] = tran[i][j]/np.sum(tran[i][j])
tran = tran.astype(np.float32)

transition = np.array([np.matmul(a,b) for a,b in zip(policy, tran)])

rewards = np.random.random([400, 400]).astype(np.float32)




resu = {}
resu['sgd'] = {}
resu['sgd']['obj']=[]
resu['sgd']['grads'] = []
resu['spider0'] = {}
resu['spider0']['obj'] = []
resu['spider0']['grads'] = []
resu['spider0']['norm'] = []
resu['spider1'] = {}
resu['spider1']['obj'] = []
resu['spider1']['grads'] = []
resu['spider1']['norm'] = []
resu['svrg'] = {}
resu['svrg']['obj'] = []
resu['svrg']['grads'] = []
resu['svrg']['norm'] = []
resu['scgd'] = {}
resu['scgd']['obj'] = []
resu['scgd']['grads'] = []



if args.layers == 1:
    hidden_sizes = [1]
elif args.layers == 2:
    hidden_sizes = [args.hidden_size, 1]
else:
    hidden_sizes = [1024, 512, 1]

class Model(object):
    def __init__(self, hidden_sizes=hidden_sizes, activation=args.activation, name='modell'):
        with tf.variable_scope(name) as scope:
            #pdb.set_trace()
            self.params = create_mlp_log(x, hidden_sizes,  name='mlp')
            self.mid_preds = forward_mlp_log(x, self.params, hidden_sizes=hidden_sizes, hidden_activation=activation,name='xmlp')
            self.preds = forward_mlp_log(x1, self.params, hidden_sizes = hidden_sizes, hidden_activation=activation, name='x1mlp')
            self.est_preds = tf.matmul(transi, self.mid_preds) + tf.reduce_sum(tf.multiply(transi, r), axis=1, keepdims=True) * 400/size[0]
            self.out_g = tf.concat([self.preds, self.est_preds], axis=0)

            self.inp_g = create_params_as(self.out_g, name='inp_g')[0]
            self.mid_g = tf.reshape(self.inp_g, [2, -1])
            #pdb.set_trace()
            mid_g = tf.gather(tf.transpose(self.mid_g), index)
            self.out_f = tf.reduce_sum(tf.square(mid_g[:,0] - mid_g[:,1]))*400/size[1]

            self.loss_g = tf.reshape(self.out_g, [2, -1])
            self.loss = tf.reduce_sum(tf.square(self.loss_g[0,:] - self.loss_g[1,:]))*400/size[1]

class EvalModel(object):
    def __init__(self, params, hidden_sizes=hidden_sizes, activation=args.activation,name='eval_model'):
        with tf.variable_scope(name) as scope:
            self.params = params
            self.preds = forward_mlp_log(trX, self.params, hidden_sizes=hidden_sizes, hidden_activation=activation, name='evalmlp')
            self.est_preds = tf.matmul(transition, self.preds) + tf.reduce_sum(tf.multiply(transition, rewards), axis=1, keepdims=True)

            self.out = tf.concat([self.preds, self.est_preds], axis=1)
            self.g = tf.reshape(self.out, [2, -1])
            self.loss = tf.reduce_sum(tf.square(self.g[0,:] - self.g[1,:]))


s1a = args.batch_size
s1b = args.batch_size
sess = tf.Session()
init = tf.global_variables_initializer()
def run_scgd(scgdopt, sess, init):
    sess.run(init)
    grads = 0
    grad_list = np.zeros(args.epochs)
    obj_list = np.zeros(args.epochs)
    for i in range(500):
        ind = np.random.choice(400, [s1a], replace=False)
        ind_f = np.random.choice(400, [s1b], replace=False)
        batchX = trX[ind]
        batch_transi = transition[ind_f, :][:, ind]
        batch_r = rewards[ind_f, :][:, ind]
        batch_x1 = trX[ind_f]
        feed_dict = {size: np.array([s1a, s1b]), x1: batch_x1, transi: batch_transi, x: batchX, r: batch_r,
                     index: np.array(range(s1a))}
        sess.run(scgdopt.update(), feed_dict=feed_dict)
        grads = grads + 2*s1a + s1b
        f = sess.run(eval.loss)
        grad_list[i] = grads
        obj_list[i] = f
        if (math.isnan(f) or math.isinf(f) or f>100):
            return (grad_list, obj_list)
        print(f)
    return (grad_list, obj_list)

def run_sgd(sgdopt, sess, init):
    sess.run(init)
    grads = 0
    grad_list = np.zeros(args.epochs)
    obj_list = np.zeros(args.epochs)
    for i in range(args.epochs):
        ind = np.random.choice(400, [s1a], replace=False)
        ind_f = np.random.choice(400, [s1b], replace=False)
        batchX = trX[ind]
        batch_transi = transition[ind_f, :][:,ind]
        batch_r = rewards[ind_f, :][:,ind]
        batch_x1 = trX[ind_f]
        feed_dict = {size: np.array([s1a,s1b]), x1: batch_x1, transi: batch_transi, x: batchX, r: batch_r, index: np.array(range(s1a))}
        sess.run(sgdopt, feed_dict=feed_dict)
        grads = grads + 2 * s1a + s1b
        f = sess.run(eval.loss)
        grad_list[i] = grads
        obj_list[i] = f
        if (math.isnan(f) or math.isinf(f)):
            return grad_list, obj_list
        print(f)
    return grad_list, obj_list

def run_spider(optimizer,sess, init, option=0):
    outer_op = optimizer.upd()
    if option == 0:
        inner_op = optimizer.minimize()
    else:
        inner_op = optimizer.minimize1()
    sess.run(init)
    grads = 0
    grad_list = np.zeros(args.epochs)
    obj_list = np.zeros(args.epochs)
    norm_list = np.zeros(args.epochs)
    for i in range(args.epochs):
        ind = np.random.choice(400, [s1a], replace=False)
        ind_f = np.random.choice(400, [s1b], replace=False)
        batchX = trX[ind]
        batch_transi = transition[ind_f,:][:, ind]
        batch_r = rewards[ind_f,:][:, ind]
        batch_x1 = trX[ind_f]
        feed_dict={size: np.array([s1a, s1b]), x1: batch_x1, transi: batch_transi, x: batchX, r:batch_r, index:np.array(range(s1a))}
        opt = sess.run(outer_op, feed_dict=feed_dict)
        [opt, g_norm, f] = sess.run([outer_op, tf.norm(optimizer.param_updates, ord=2), eval.loss], feed_dict=feed_dict)
        # g_norm = sess.run(tf.norm(optimizer1.param_updates, ord=2), feed_dict=feed_dict)
        # f = sess.run(eval.loss, feed_dict=feed_dict)
        grads = grads + 2 * s1a + s1b
        grad_list[i] = grads
        obj_list[i] = f
        norm_list[i] = g_norm
        print(f)
        if (math.isnan(f) or math.isinf(f)):
            return (grad_list, obj_list, norm_list)
        for j in range(args.minibatchsize):
            ind = np.random.choice(s1a, [s2a], replace=False)
            ind_f = np.random.choice(s1b, [s2b], replace=False)
            batchXX = batchX[ind]
            batch_transitransi = batch_transi[:,ind]
            batch_rr = batch_r[:,ind]
            feed_dict={size: np.array([s2a, s2b]), x1:batch_x1, transi:batch_transitransi, x:batchXX, r:batch_rr, index:ind_f}
            opt = sess.run(inner_op, feed_dict=feed_dict)
            #g_norm = sess.run(tf.norm(optimizer.param_updates, ord=2), feed_dict=feed_dict)
            grads = grads + 4*s2a + 2 * s2b
            f = sess.run(eval.loss)
    return (grad_list, obj_list, norm_list)

def run_svrg(optimizer1, sess, init):
    outer_op1 = optimizer1.upd()
    inner_op1 = optimizer1.minimize()
    sess.run(init)
    grads = 0
    grad_list = np.zeros(args.epochs)
    obj_list = np.zeros(args.epochs)
    norm_list = np.zeros(args.epochs)
    for i in range(args.epochs):
        ind = np.random.choice(400, [s1a], replace=False)
        ind_f = np.random.choice(400, [s1b], replace=False)
        batchX = trX[ind]
        batch_transi = transition[ind_f,:][:, ind]
        batch_r = rewards[ind_f,:][:, ind]
        batch_x1 = trX[ind_f]
        feed_dict={size: np.array([s1a, s1b]), x1: batch_x1, transi: batch_transi, x: batchX, r:batch_r, index:np.array(range(s1a))}
        opt = sess.run(outer_op1, feed_dict=feed_dict)
        [opt,g_norm, f] = sess.run([outer_op1, tf.norm(optimizer1.param_updates, ord=2), eval.loss], feed_dict=feed_dict)
        #g_norm = sess.run(tf.norm(optimizer1.param_updates, ord=2), feed_dict=feed_dict)
        #f = sess.run(eval.loss, feed_dict=feed_dict)
        grad_list[i] = grads
        #pdb.set_trace()
        obj_list[i] = f
        norm_list[i] = g_norm
        grads = grads + 2 * s1a + s1b
        print(f)
        if (math.isnan(f) or math.isinf(f)):
            return (grad_list, obj_list, norm_list)
        for j in range(args.minibatchsize):
            ind = np.random.choice(s1a, [s2a], replace=False)
            ind_f = np.random.choice(s1b, [s2b], replace=False)
            batchXX = batchX[ind]
            batch_transitransi = batch_transi[:,ind]
            batch_rr = batch_r[:,ind]
            feed_dict={size: np.array([s2a, s2b]), x1:batch_x1, transi:batch_transitransi, x:batchXX, r:batch_rr, index:ind_f}
            opt = sess.run(inner_op1, feed_dict=feed_dict)
            grads = grads + 4 * s2a + 2 * s2b
            #g_norm = sess.run(tf.norm(optimizer1.param_updates, ord=2), feed_dict=feed_dict)
            #grads = grads + 4 * s2a + 2 * s2b
            #f = sess.run(eval.loss)

    return (grad_list, obj_list, norm_list)
lr_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
bs_list = [100, 50, 20, 10]
mini_list = [10, 5]
for lr in [2e-3]:
#    for bs in bs_list:
    bs = 50
    tf.reset_default_graph()
    args.batch_size = bs
    args.learning_rate = lr
    s1a = args.batch_size
    s1b = args.batch_size
    x = tf.placeholder(tf.float32, [None, 201], name='x')
    transi = tf.placeholder(tf.float32, [s1b, None], name='transi')
    r = tf.placeholder(tf.float32, [s1b, None], name='rewards')
    x1 = tf.placeholder(tf.float32, [s1b, 201], name='x1')
    # y = tf.placeholder(tf.float32, [None, 10])
    size = tf.placeholder(tf.float32, [2], name='size')
    index = tf.placeholder(tf.int32, [None], name='index')
    #pdb.set_trace()
    model = Model(name='model_scgd')#

    eval = EvalModel(model.params)

    scgdopt = scgd_optimizer(args.learning_rate, model)
    # pdb.set_trace()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    grad, obj = run_scgd(scgdopt, sess, init)
    resu['scgd']['grads'].append(grad)
    resu['scgd']['obj'].append(obj)
pdb.set_trace()

with open(Filename+'scgd', 'wb') as f:
    pickle.dump(resu, f)

#print(Filename)
for lr in lr_list:
    for bs in bs_list:
        tf.reset_default_graph()
        args.batch_size = bs
        args.learning_rate = lr
        s1a = args.batch_size
        s1b = args.batch_size
        x = tf.placeholder(tf.float32, [None, 201], name='x')
        transi = tf.placeholder(tf.float32, [args.batch_size, None], name='transi')
        r = tf.placeholder(tf.float32, [args.batch_size, None], name='rewards')
        x1 = tf.placeholder(tf.float32, [args.batch_size, 201], name='x1')
        # y = tf.placeholder(tf.float32, [None, 10])
        size = tf.placeholder(tf.float32, [2], name='size')
        index = tf.placeholder(tf.int32, [None], name='index')

        model = Model(name='model_ori')
        control_model = Model(name='model_control')

        eval = EvalModel(model.params)
        pdb.set_trace()

        sgdopt = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(model.loss)
        # pdb.set_trace()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        grad, obj = run_sgd(sgdopt, sess, init)
        resu['sgd']['grads'].append(grad)
        resu['sgd']['obj'].append(obj)
pdb.set_trace()

with open(Filename+'sgd', 'wb') as f:
    pickle.dump(resu, f)

for lr in lr_list[1:5]:
    for bs in bs_list[2:]:
        for mini in mini_list:
            tf.reset_default_graph()
            args.batch_size = bs
            args.learning_rate = lr
            args.minibatchsize = mini
            s1a = args.batch_size
            s1b = args.batch_size
            x = tf.placeholder(tf.float32, [None, 201], name='x')
            transi = tf.placeholder(tf.float32, [args.batch_size, None], name='transi')
            r = tf.placeholder(tf.float32, [args.batch_size, None], name='rewards')
            x1 = tf.placeholder(tf.float32, [args.batch_size, 201], name='x1')
            # y = tf.placeholder(tf.float32, [None, 10])
            size = tf.placeholder(tf.float32, [2], name='size')
            index = tf.placeholder(tf.int32, [None], name='index')

            model = Model(name='model_ori')
            control_model = Model(name='model_control')

            eval = EvalModel(model.params)

            optimizer = spider_scgd_optimizer(args.learning_rate, model, control_model)
            sess = tf.Session()
            init = tf.global_variables_initializer()

            grad, obj, norm = run_spider(optimizer, sess, init, option=1)
            resu['spider1']['grads'].append(grad)
            resu['spider1']['obj'].append(obj)
            resu['spider1']['norm'].append(norm)

for lr in lr_list[1:5]:
    for bs in bs_list[2:]:
        for mini in mini_list:
            tf.reset_default_graph()
            args.batch_size = bs
            args.learning_rate = lr
            args.minibatchsize = mini
            s1a = args.batch_size
            s1b = args.batch_size
            x = tf.placeholder(tf.float32, [None, 201], name='x')
            transi = tf.placeholder(tf.float32, [args.batch_size, None], name='transi')
            r = tf.placeholder(tf.float32, [args.batch_size, None], name='rewards')
            x1 = tf.placeholder(tf.float32, [args.batch_size, 201], name='x1')
            # y = tf.placeholder(tf.float32, [None, 10])
            size = tf.placeholder(tf.float32, [2], name='size')
            index = tf.placeholder(tf.int32, [None], name='index')

            model = Model(name='model_ori')
            control_model = Model(name='model_control')

            eval = EvalModel(model.params)

            optimizer = spider_scgd_optimizer(args.learning_rate, model, control_model)
            sess = tf.Session()
            init = tf.global_variables_initializer()

            grad, obj, norm = run_spider(optimizer, sess, init, option=0)
            resu['spider0']['grads'].append(grad)
            resu['spider0']['obj'].append(obj)
            resu['spider0']['norm'].append(norm)

for lr in lr_list[3:]:
    for bs in bs_list[2:]:
        for mini in mini_list:
            print('new')
            tf.reset_default_graph()
            args.batch_size = bs
            args.learning_rate = lr
            args.minibatchsize = mini
            s1a = args.batch_size
            s1b = args.batch_size
            x = tf.placeholder(tf.float32, [None, 201], name='x')
            transi = tf.placeholder(tf.float32, [args.batch_size, None], name='transi')
            r = tf.placeholder(tf.float32, [args.batch_size, None], name='rewards')
            x1 = tf.placeholder(tf.float32, [args.batch_size, 201], name='x1')
            # y = tf.placeholder(tf.float32, [None, 10])
            size = tf.placeholder(tf.float32, [2], name='size')
            index = tf.placeholder(tf.int32, [None], name='index')

            model = Model(name='model_ori')
            control_model = Model(name='model_control')

            eval = EvalModel(model.params)
            optimizer1 = vrscpg_optimizer(args.learning_rate, model, control_model)
            sess = tf.Session()
            init = tf.global_variables_initializer()

            #pdb.set_trace()

            grad, obj, norm = run_svrg(optimizer1, sess, init)
            resu['svrg']['grads'].append(grad)
            resu['svrg']['obj'].append(obj)
            resu['svrg']['norm'].append(norm)

with open(Filename, 'wb') as f:
    pickle.dump(resu, f)

print(Filename)