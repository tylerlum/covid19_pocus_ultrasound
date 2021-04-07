import tensorflow as tf
import numpy as np


def evidential_loss(nb_classes):
    # evidential loss function
    def KL(alpha, K):
        beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

        KL = (tf.reduce_sum((alpha - beta)*(tf.math.digamma(alpha)-tf.math.digamma(S_alpha)), axis=1, keepdims=True) +
              tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True) +
              tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True)))
        return KL

    def loss_eq5(actual, pred, K, global_step, annealing_step):
        p = actual
        p = tf.dtypes.cast(p, tf.float32)
        alpha = pred + 1.
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        loglikelihood = tf.reduce_sum((p-(alpha/S))**2, axis=1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
        KL_reg = tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1, K)
        return loglikelihood + KL_reg

    ev_loss = (
        lambda actual, pred: tf.reduce_mean(loss_eq5(actual, pred, nb_classes, 1, 100))
    )
    return ev_loss
