import tensorflow as tf
ratio = 2
def passby(inputs):
	o1 = tf.layers.conv2d(inputs = inputs, filters = 128,kernel_size = 1, padding = 'same')
	ro1 = tf.nn.relu(o1)
	o2 = tf.layers.conv2d(inputs = ro1, filters = 256,kernel_size = 3, padding = 'same')
	ro2 = tf.nn.relu(o2)
	o3 = tf.layers.conv2d(inputs = ro2, filters = 256,kernel_size = 1, padding = 'same')
	return o3+inputs

def stacked_hourglass(inputs):

	pm1 = passby(inputs)
	m1 = tf.nn.max_pool(pm1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	pm2 = passby(m1)
	m2 = tf.nn.max_pool(pm2,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	pm3 = passby(m2)
	m3 = tf.nn.max_pool(pm3,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	pm4 = passby(m3)
	m4 = tf.nn.max_pool(pm4,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	pm5 = passby(m4)

	bm5 = passby(pm5)

	b1 = passby(pm1)
	b2 = passby(pm2)
	b3 = passby(pm3)
	b4 = passby(pm4)

	bm4 = tf.image.resize_nearest_neighbor(passby(bm5),[16*ratio,16*ratio])+b4
	bm3 = tf.image.resize_nearest_neighbor(passby(bm4),[32*ratio,32*ratio])+b3
	bm2 = tf.image.resize_nearest_neighbor(passby(bm3),[64*ratio,64*ratio])+b2
	bm1 = tf.image.resize_nearest_neighbor(passby(bm2),[128*ratio,128*ratio])+b1

	return passby(bm1)

