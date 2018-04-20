import tensorflow as tf
import tensorflow.contrib.layers as layers


def atari_model():
    model = cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512])
    return model


def _mlp(hiddens, inpt, scope, reuse, layer_norm):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)

    return out


def mlp(hiddens, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    var_func: function
        representing the VaR of the CVaR DQN algorithm
    cvar_func: function
        representing the CVaRy of the CVaR DQN algorithm
    """

    def last_layer(name, hiddens, inpt, num_actions, nb_atoms, scope,
                   reuse_main=False, reuse_last=False, layer_norm=False):
        out = _mlp(hiddens, inpt, scope + '/net', reuse_main, layer_norm)
        with tf.variable_scope('{}/{}'.format(scope, name), reuse=reuse_last):
            out = layers.fully_connected(out, num_outputs=num_actions * nb_atoms, activation_fn=None)
            out = tf.reshape(out, shape=[-1, num_actions, nb_atoms], name='out')
        return out

    var_func = lambda *args, **kwargs: last_layer('var', hiddens, layer_norm=layer_norm, *args, **kwargs)
    cvar_func = lambda *args, **kwargs: last_layer('cvar', hiddens, layer_norm=layer_norm, *args, **kwargs)

    return var_func, cvar_func


def _cnn_to_mlp(convs, hiddens, inpt, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                out = tf.nn.relu(action_out)

        return out


def cnn_to_mlp(convs, hiddens, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    var_func: function
        representing the VaR of the CVaR DQN algorithm
    cvar_func: function
        representing the CVaRy of the CVaR DQN algorithm
    """

    def last_layer(name, convs, hiddens, inpt, num_actions, nb_atoms, scope,
                   reuse_main=False, reuse_last=False, layer_norm=False):
        out = _cnn_to_mlp(convs, hiddens, inpt, scope + '/net', reuse_main, layer_norm)
        with tf.variable_scope('{}/{}'.format(scope, name), reuse=reuse_last):
            out = layers.fully_connected(out, num_outputs=num_actions * nb_atoms, activation_fn=None)
            out = tf.reshape(out, shape=[-1, num_actions, nb_atoms], name='out')
        return out

    var_func = lambda *args, **kwargs: last_layer('var', convs, hiddens, layer_norm=layer_norm, *args, **kwargs)
    cvar_func = lambda *args, **kwargs: last_layer('cvar', convs, hiddens, layer_norm=layer_norm, *args, **kwargs)

    return var_func, cvar_func