import tensorflow as tf

def last(inputs,GT_center,GT_wh):
    after_p = tf.layers.average_pooling2d(inputs,[256,256],1)
    squeeze = tf.reshape(after_p,[-1,256])
    fc1 = tf.layers.dense(squeeze,1024,activation = tf.nn.elu)
    out = tf.layers.dense(fc1,5)
    out_center = out[:,0:3]
    out_wh = out[:,3:]
    loss_center = smooth_l1(out_center,GT_center)
    loss_wh = 0*smooth_l1(out_wh,GT_wh)
    return loss_wh+loss_center,out_center,out_wh

def smooth_l1(A,B):
    return tf.losses.huber_loss(A,B)