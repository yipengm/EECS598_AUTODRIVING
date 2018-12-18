import tensorflow as tf
from SHnet import stacked_hourglass as SHnet
from last_loc import last
import numpy as np
from PIL import Image
fo = open("loc.csv",'w')
import matplotlib.pyplot as plt

image_size = 256
regularize_para = 0.4

img_rgb_in = tf.placeholder(tf.float32, [None, image_size*image_size*3])

img_depth_in = tf.placeholder(tf.float32, [None, image_size*image_size])

img_rgb = tf.reshape(img_rgb_in,[-1, image_size, image_size, 3])

img_depth = tf.reshape(img_depth_in,[-1,image_size,image_size,1])

inputs = tf.concat([img_rgb,img_depth],3)

center_in = tf.placeholder(tf.float32,[None,3])

GT_wh = tf.placeholder(tf.float32,[None,2])

img2dark = tf.layers.conv2d(inputs = inputs, filters = 256,kernel_size = 1, padding = 'same')
with tf.name_scope("SH2"):
	dark_out1 = SHnet(img2dark)

with tf.name_scope("SH3"):
	dark_out = SHnet(dark_out1)

#center_loss1,class_loss1,_ = last(dark_out1,center_in,label_in)
loss,pred_center,_ = last(dark_out,center_in,GT_wh)
initial_learning_rate = 5e-5

train_op = tf.train.AdamOptimizer(learning_rate = initial_learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver1 = tf.train.Saver()
writer = tf.summary.FileWriter("logs/loc",  tf.get_default_graph())
saver1.restore(sess,'./ckpt_loc1/other.ckpt')

im_name_array = []
with open('TEST') as my_file:
	for line in my_file:
		im_name_array.append(line)
epochs = 1
data_num = len(im_name_array)
batch_size = 1
correct = 0
wrong = 0

for epoch in range(epochs):
	index_rand = range(data_num)
	for j in range(int(data_num/batch_size)):
		image_data = np.zeros([batch_size,image_size,image_size,3])
		depth_data = np.zeros([batch_size,image_size,image_size])
		label_data = np.zeros([batch_size,2])
		label_check = np.zeros([batch_size,1])
		for i in range(batch_size):
			index = i+j*batch_size
			#name_base = "./deploy/trainval/06acf647-e2e9-4562-aa42-eed217e5bd84/0000"
			name_base = "./deploy/test/" + im_name_array[index_rand[index]][0:-3]
			rgb_path = name_base+"_image_256.jpg"
			depth_path = name_base+"_depth_256.jpg"
			rgb = Image.open(rgb_path)
			dep = Image.open(depth_path)
			image_data[i,:,:,:] = np.array(rgb).astype(float)
			depth_data[i,:,:] = np.array(dep).astype(float)

			#label = 0
			label = int(im_name_array[index_rand[index]][-2])
			if label == 0:
				label = 1
			else:
				label -=1
			one_hot_label = np.zeros(2)
			one_hot_label[label] = 1
			label_data[i,:] = one_hot_label
			label_check[i,0] = label
			proj = np.fromfile(name_base+'_proj.bin', dtype=np.float32)
			proj.resize([3, 4])

		depth_data_in = np.reshape(depth_data,[batch_size,256*256])/255
		image_data_in = np.reshape(image_data,[batch_size,256*256*3])/255
		pred_center_ = sess.run([pred_center],feed_dict={img_rgb_in: image_data_in, img_depth_in: depth_data_in, GT_wh:np.zeros([batch_size,2],np.float)})

		print("##########")
		print('Epoch',epoch,'|Step:', j)
		print('pred_center',pred_center_[0])
		print("##########")

		xyz_p = np.reshape(np.array(pred_center_[0]),[3])
		xyz_p[0:2] = xyz_p[0:2]*xyz_p[2]
		xyz_p[0] = 1914/256*xyz_p[0]
		xyz_p[1] = 1052/256*xyz_p[1]
		F_M = proj[0:3,0:3]
		xyz_o = np.linalg.inv(F_M)@(xyz_p - proj[:,3])
		print("##########")
		print(xyz_o)
		print(name_base[14:] + "/x," + str(xyz_o[0]),file = fo)
		print(name_base[14:] + "/y," + str(xyz_o[1]), file=fo)
		print(name_base[14:] + "/z," + str(xyz_o[2]), file=fo)
		print("##########")









