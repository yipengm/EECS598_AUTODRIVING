import tensorflow as tf

def last(inputs,GT_center,GT_label):
    after_p = tf.layers.average_pooling2d(inputs,[256,256],1)
    squeeze = tf.reshape(after_p,[-1,256])
    fc1 = tf.layers.dense(squeeze,1024,activation = tf.nn.elu)
    out = tf.layers.dense(fc1,4)
    class_pred_0 = tf.reshape(out[:,0],[-1,1])
    class_pred = tf.layers.batch_normalization(class_pred_0,training = 1,axis = -1)
    x_pred = tf.reshape(out[:,1],[-1,1])
    y_pred = tf.reshape(out[:,2],[-1,1])
    z_pred = tf.reshape(out[:,3],[-1,1])
    center_pred = tf.concat([x_pred,y_pred,z_pred],1)
    center_loss_expand = smooth_l1(center_pred,GT_center)

    class_poss1 = tf.nn.sigmoid(class_pred)
    class_poss2 = 1-class_poss1
    class_poss = tf.concat([class_poss1,class_poss2],1)

    class_loss = tf.reduce_mean(-tf.reduce_sum(GT_label*tf.log(class_poss),1))
    center_loss = tf.reduce_sum(center_loss_expand)/2
    return center_loss,class_loss,out,class_pred

def smooth_l1(A,B):
    return tf.losses.huber_loss(A,B)