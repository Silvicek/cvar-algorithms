"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    alpha: float
        Action is picked to maximize CVaR_alpha
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative no update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes CVaR TD error:

        TODO: describe error

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in CVaR Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized P function to the target P function.
    TODO: ???
    In distributional RL we actually optimize the following error:

        ThTz(P') * log(P)

    Where P' is lagging behind P to stabilize the learning.

"""
import tensorflow as tf
import baselines.common.tf_util as U


def pick_actions(cvar_values):
    """
    Select actions based on optimal CVaR value for each atom.
    Parameters
    ----------
    cvar_values: (?, actions, nb_atoms)

    Returns
    -------
    (?, nb_atoms)
    """
    deterministic_actions = tf.argmax(cvar_values, axis=-1, output_type=tf.int32)
    return deterministic_actions

debug_expressions = []

def pick_action(cvar_values, alpha, nb_atoms):
    """
    Pick a single action based on CVaR_alpha.
    Assumes linearly spaced atoms.

    Parameters
    ----------
    cvar_values: (?, actions, nb_atoms)

    Returns
    -------
    (?,)

    """

    ix_f = alpha*nb_atoms - 1
    ix_int = tf.cast(tf.floor(ix_f), tf.int32)
    portion = ix_f - tf.cast(ix_int, tf.float32)
    # special case if alpha=1
    ix_next = tf.cond(tf.equal(alpha, tf.constant(1, tf.float32)), lambda: ix_int, lambda: ix_int+1, name='ix_next')

    cvar_alpha_std = cvar_values[:, :, ix_int] * (1-portion) + cvar_values[:, :, ix_next] * portion

    # if alpha is before first atom
    cvar_alpha_zero = cvar_values[:, :, ix_next] * portion

    cvar_alpha = tf.cond(tf.less(alpha, 1/nb_atoms), lambda: cvar_alpha_zero, lambda: cvar_alpha_std)

    return tf.argmax(cvar_alpha, axis=-1, output_type=tf.int32)


def build_act(make_obs_ph, cvar_func, var_func, num_actions, nb_atoms, scope="cvar_dqn", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    cvar_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    nb_atoms: int
        number of linearly-spaced atoms
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        # alpha in cvar_alpha
        alpha_ph = U.ensure_tf_input(tf.placeholder(tf.float32, (), name="alpha"))

        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        # eps in epsilon-greedy
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        cvar_values = cvar_func(observations_ph.get(), num_actions, nb_atoms, scope="out_func")
        # keep here for plotting
        var_values = var_func(observations_ph.get(), num_actions, nb_atoms, scope="out_func",
                              reuse_main=True, reuse_last=False)

        deterministic_actions = pick_action(cvar_values, alpha_ph.get(), nb_atoms)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int32)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = U.function(inputs=[observations_ph, alpha_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])

        return act


def extract_distribution(y_cvar, nb_atoms):
    """ Convert yC -> underlying distribution.
        y_cvar: (?, nb_atoms)
    """

    dist_cropped = y_cvar[:, 1:] - y_cvar[:, :-1]
    dist = tf.concat((y_cvar[:, 0, None], dist_cropped), axis=1) * nb_atoms
    return dist


def build_train(make_obs_ph, var_func, cvar_func, num_actions, nb_atoms, optimizer, grad_norm_clipping=None, gamma=1.0,
                scope="cvar_dqn", reuse=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    var_func: (tf.Variable, int, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            nb_atoms: int
                number of atoms
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    cvar_func: (tf.Variable, int, str, bool) -> tf.Variable
        see var_func
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f = build_act(make_obs_ph, cvar_func, var_func, num_actions, nb_atoms, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        # atoms
        y = tf.range(1, nb_atoms + 1, dtype=tf.float32, name='y') * 1. / nb_atoms

        # ------------------------------- Core networks ---------------------------------
        # var network
        var_t = var_func(obs_t_input.get(), num_actions, nb_atoms, scope="out_func",
                         reuse_main=True, reuse_last=True) # reuse from act

        # vars for actions which we know were selected in the given state.
        var_t_selected = gather_along_second_axis(var_t, act_t_ph)
        var_t_selected.set_shape([None, nb_atoms])

        # cvar network
        cvar_t = cvar_func(obs_t_input.get(), num_actions, nb_atoms, scope="out_func",
                           reuse_main=True, reuse_last=True)  # reuse from act

        # cvars for actions which we know were selected in the given state.
        cvar_t_selected = gather_along_second_axis(cvar_t, act_t_ph)
        cvar_t_selected.set_shape([None, nb_atoms])

        # target cvar network
        cvar_tp1 = cvar_func(obs_tp1_input.get(), num_actions, nb_atoms, scope="target_cvar_func")

        # extract variables
        joint_variables = U.scope_vars(U.absolute_scope_name("out_func/net"))
        var_variables = U.scope_vars(U.absolute_scope_name("out_func/var"))
        cvar_variables = U.scope_vars(U.absolute_scope_name("out_func/cvar"))
        target_cvar_func_variables = U.scope_vars(U.absolute_scope_name("target_cvar_func"))

        # -------------------------------------------------------------------------------

        # ----------------------------- Extract distribution ----------------------------
        # construct a new cvar with different actions for each atom
        cvar_tp1_star = tf.reduce_max(cvar_tp1, axis=1)
        cvar_tp1_star.set_shape([None, nb_atoms])
        # construct a distribution from the new cvar
        ycvar_tp1_star = cvar_tp1_star * y
        dist_tp1_star_ = extract_distribution(ycvar_tp1_star, nb_atoms)

        # apply done mask
        dist_tp1_star = tf.einsum('ij,i->ij', dist_tp1_star_, 1. - done_mask_ph)

        # Td = r + gamma * dist
        dist_target = tf.identity(rew_t_ph[:, None] + gamma * dist_tp1_star, name='dist_target')
        # dist is always non-differentiable
        dist_target = tf.stop_gradient(dist_target)

        # -------------------------------------------------------------------------------

        # ---------------------------------- VaR loss -----------------------------------

        td_error = dist_target[:, :, None] - var_t_selected[:, None, :]
        # td_error[0]=
        #  [[Td1-v1 Td1-v2 ... Td1-vn]
        #   [Td2-v1 Td2-v2 ... Td2-vn]
        #   [...                     ]
        #   [Tdn-v1 Tdn-v2 ... Tdn-vn]]

        negative_indicator = tf.cast(td_error < 0, tf.float32)

        var_weights = tf.stop_gradient(y - negative_indicator)  # XXX: stop gradient?
        quantile_loss = var_weights * td_error

        var_error = tf.reduce_mean(quantile_loss)
        # -------------------------------------------------------------------------------

        # ---------------------------------- CVaR loss ----------------------------------
        # Minimizing the MSE of:
        # V_i + 1/y_i(Td_j - V_i)^- - C_i

        min_target_diff = negative_indicator / y * tf.stop_gradient(td_error)
        cvar_loss = tf.stop_gradient(var_t_selected)[:, None, :] + min_target_diff - cvar_t_selected[:, None, :]

        cvar_error = tf.reduce_mean(tf.square(cvar_loss))

        # -------------------------------------------------------------------------------

        # ------------------------------- Finalizing ------------------------------------

        error = var_error + cvar_error
        # compute optimization op (potentially with gradient clipping)
        var_list = [joint_variables, var_variables, cvar_variables]
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer, error, var_list, clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(error, var_list=var_list)

        # update_target_fn will be called periodically to copy cvar network to target cvar network
        # Note: var has no target
        update_target_expr = []
        for cvar_variable, target_cvar_variable in zip(sorted(joint_variables+cvar_variables, key=lambda v: v.name),
                                                       sorted(target_cvar_func_variables, key=lambda v: v.name)):
            update_target_expr.append(target_cvar_variable.assign(cvar_variable))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        # -------------------------------------------------------------------------------

        # --------------------------------- Debug ---------------------------------------
        a = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], var_t_selected)
        b = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], cvar_t_selected)
        # c = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], big_dist_target*y)
        # b = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], var_t)
        # c = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], negative_indicator)
        # d = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], big_yc_target)
        # e = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], cvar_t)
        # f = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], cvar_loss)
        # atoms = U.function([obs_tp1_input], atoms)
        # -------------------------------------------------------------------------------

        return act_f, train, update_target, [a, b]


def gather_along_second_axis(data, indices):
    batch_offset = tf.range(0, tf.shape(data)[0])
    flat_indices = tf.stack([batch_offset, indices], axis=1)
    return tf.gather_nd(data, flat_indices)
