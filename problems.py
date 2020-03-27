import tensorflow as tf
import pdb
import numpy as np
def create_mlp_log(input, hidden_sizes, input_shape=None, name='mlp', reuse=False, regularizer=tf.contrib.layers.l2_regularizer(1.0)):
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.contrib.layers.xavier_initializer()
    #w_init = tf.constant_initializer(0.0)
    #b_init = tf.constant_initializer(0.0)
    all_params=[]
    idx=0
    if input_shape is None:
        input_shape = input.get_shape()
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        curr_size = input_shape[1]
        for hidden_size in hidden_sizes:
            W = tf.get_variable('w'+str(idx), shape=(curr_size, hidden_size), dtype=tf.float32, initializer=w_init, regularizer=regularizer)
            b = tf.get_variable('b'+str(idx), shape=(hidden_size,), dtype = tf.float32, initializer=b_init)
            curr_size = hidden_size
            all_params.append(W)
            all_params.append(b)
            idx += 1

    return all_params

def forward_mlp_log(input, all_params, hidden_sizes, hidden_activation=tf.identity, name='mlp'):
    n = len(hidden_sizes)
    cur = input
    with tf.variable_scope(name) as scope:
        #pdb.set_trace()
        for idx in range(n):
            #pdb.set_trace()
            cur = tf.matmul(cur, all_params[2 * idx])
            cur = cur + tf.expand_dims(all_params[2 * idx + 1], 0)
            cur = hidden_activation(cur)

        #pred = tf.nn.softmax(cur)
        pred = cur
    return pred


def create_params_as(params, name, reuse=False):
    param_list = []
    if type(params) is not list:
        params = [params]
    n = len(params)
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        for idx in range(n):
            xw = params[idx]
            xws = xw.get_shape()
            if idx%2==0:
                W = tf.get_variable('w'+str(int(idx/2)), shape=xws, dtype=tf.float32)

            else:
                W = tf.get_variable('b'+str(int(idx/2)), shape=xws, dtype=tf.float32)

            param_list.append(W)

    return param_list

def assign_params(params_1, params_2):
    update_ops = []
    if type(params_1) is not list:
        update_ops = params_1.assign(params_2)
        return update_ops
    if type(params_2) is not list:
        current=0
        #pdb.set_trace()
        for i in range(len(params_1)):
            param_1 = params_1[i]
            param_prod = np.prod(var_shape(param_1))
            param_2 = tf.reshape(params_2[current:(current+param_prod)], var_shape(param_1))
            op = param_1.assign(param_2)
            update_ops.append(op)
            current += param_prod
        return update_ops
    for i in range(len(params_1)):
        param_1 = params_1[i]
        param_2 = params_2[i]
        op = param_1.assign(param_2)
        update_ops.append(op)
    return update_ops

def flat_gradients(loss, var_list):
    """
    Same as tf.gradients but returns flat tensor.
    """
    grads = tf.gradients(loss, var_list)
    return tf.concat(values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(var_list, grads)], axis=0)


def vector_gradients(loss, var_list):
    #pdb.set_trace()
    resu = [flat_gradients(loss[0], var_list)]
    for i in range(1, loss.get_shape()[0]):
        print(i)
        #pdb.set_trace()
        resu = tf.concat([resu, [flat_gradients(loss[i], var_list)]], axis=0)
    #return tf.concat(values=[flat_gradients(loss[i], var_list) for i in range(loss.get_shape()[0])], axis=1)
    return resu

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out
