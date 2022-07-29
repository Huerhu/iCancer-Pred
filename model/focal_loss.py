# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:01:41 2020

@author: lwzjc
"""
import tensorflow as tf
from tensorflow.keras import backend as K


def binary_focal_loss(gamma=2, alpha=0.25):
     """
     Binary form of focal loss.
     Binary Classification
     focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
     References:

     Usage:
      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
     """
     alpha = tf.constant(alpha, dtype=tf.float32)
     gamma = tf.constant(gamma, dtype=tf.float32)
     
     def binary_focal_loss_fixed(y_true, y_pred):
      """
      y_true shape need be (None,1)
      y_pred need be compute after sigmoid
      """
      y_true = tf.cast(y_true, tf.float32)
      alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
     
      p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
      focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
      return K.mean(focal_loss)
     
     return binary_focal_loss_fixed
print(binary_focal_loss())


def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    Multi-Classification
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, shape=[len(alpha), 1], dtype=tf.float32)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -K.log(y_t)
        weight = K.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    Multi-Classification / multi-label
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -K.log(y_t)
        weight = K.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed


if __name__ == '__main__':
    print(multi_category_focal_loss2(0.25, 2)(0.2, 0.6))
    