import tensorflow as tf
import tensorflow.contrib.eager as tfe

def train_vanilla_pg_policy(policy_func, model, optimizer, inputs, targets, advantage):
  with tfe.GradientTape() as tape:
    p = policy_func(inputs, training=True)
    p = tf.clip_by_value(p, 1e-6, 0.999999)
    loss = p * tf.cast(targets, dtype=tf.float32)
    loss = tf.reshape(loss, (p.shape[0], -1))
    loss = tf.reduce_sum(loss, reduction_indices=1)
    loss = tf.log(loss)

    final_loss = -tf.reduce_mean(loss * advantage)

  grads = tape.gradient(final_loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())