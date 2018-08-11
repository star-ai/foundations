import tensorflow as tf
import tensorflow.contrib.eager as tfe

def train_vanilla_pg_policy(policy_func, model, optimizer, inputs, targets, advantage):
  with tfe.GradientTape() as tape:
    p = policy_func(inputs, training=True)
    p = tf.clip_by_value(p, 1e-6, 0.999999)
    # p = tf.log(p)
    loss = p * tf.cast(targets, dtype=tf.float32)
    loss = tf.reshape(loss, (p.shape[0], -1))
    loss = tf.reduce_sum(loss, reduction_indices=1)
    loss = tf.log(loss) # really log should be done before p * target, however target tends to be 1 hot so it's ok

    final_loss = -tf.reduce_mean(loss * advantage)

  grads = tape.gradient(final_loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

def train_vanilla_ddpg_policy(policy_func, q_func, model, optimizer, state):
  with tfe.GradientTape() as tape:
    action = policy_func(state, training=False)
    with tfe.GradientTape() as dqda_tape:
      dqda_tape.watch(action)
      q = q_func([state, action])
    dqda = dqda_tape.gradient(q, action)
    loss = - dqda * action
  grads = tape.gradient(loss, model.variables)

  optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())


def train_vanilla_pg_value(value_func, model, optimizer, inputs, targets):
  with tfe.GradientTape() as tape:
    v = value_func(inputs, training=True)
    loss = tf.losses.mean_squared_error(targets, v)
    # print("V Loss:", loss)

  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())