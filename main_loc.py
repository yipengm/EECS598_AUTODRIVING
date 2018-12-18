import tensorflow as tf
from SHnet import stacked_hourglass as SHnet
from last_loc import last
import numpy as np
from PIL import Image
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


im_name_array = []
with open('IN') as my_file:
	for line in my_file:
		im_name_array.append(line)
epochs = 10
data_num = len(im_name_array)
batch_size = 4
correct = 0
wrong = 0

for epoch in range(epochs):
	index_rand = np.random.permutation(data_num)
	for j in range(int(data_num/batch_size)):
		image_data = np.zeros([batch_size,image_size,image_size,3])
		depth_data = np.zeros([batch_size,image_size,image_size])
		label_data = np.zeros([batch_size,2])
		label_check = np.zeros([batch_size,1])
		centre_data = np.zeros([batch_size,3])
		for i in range(batch_size):
			index = i+j*batch_size
			#name_base = "./deploy/trainval/06acf647-e2e9-4562-aa42-eed217e5bd84/0000"
			name_base = "./deploy/trainval/" + im_name_array[index_rand[index]][0:-3]
			rgb_path = name_base+"_image_256.jpg"
			depth_path = name_base+"_depth_256.jpg"
			rgb = Image.open(rgb_path)
			dep = Image.open(depth_path)
			image_data[i,:,:,:] = np.array(rgb).astype(float)
			depth_data[i,:,:] = np.array(dep).astype(float)
			bbox_path = name_base+"_bbox.bin"
			centre = np.fromfile(bbox_path, dtype=np.float32)

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
			centre_vec = np.zeros([4],np.float32)
			centre_vec[0:3] = centre[3:6]
			centre_vec[3] = 1
			proj = np.fromfile(name_base+'_proj.bin', dtype=np.float32)
			proj.resize([3, 4])
			center_projected = proj @ centre_vec
			centre_data[i, 0] = center_projected[0]/center_projected[2]/1914*256
			centre_data[i, 1] = center_projected[1]/center_projected[2]/1052*256
			centre_data[i, 2] = center_projected[2]

		depth_data_in = np.reshape(depth_data,[batch_size,256*256])/255
		image_data_in = np.reshape(image_data,[batch_size,256*256*3])/255
		_,loss_,pred_center_ = sess.run([train_op,loss,pred_center],feed_dict={img_rgb_in: image_data_in, img_depth_in: depth_data_in, GT_wh:np.zeros([batch_size,2],np.float),center_in:centre_data})

		if j%1==0:
			print("##########")
			print('Epoch',epoch,'|Step:', j,'|total_loss',loss_)
			print('centre_data',centre_data[0,:])
			print('pred_center',pred_center_[0,:])
			print("##########")
		if j % 1000 == 0:
			print("##########saving############")
			saver1.save(sess, './ckpt_loc/other.ckpt')
			print("########end saving##########")

		if j % 10000 == 0:
			saver1.save(sess, './ckpt_loc1/other_'+str(epoch)+'_'+str(j)+'.ckpt')

##INspect
'''
pic_0 = image_data[0, :, :, :]/255
pic_1 = image_data[1, :, :, :]/255

label_0 = label_data[0,:]
label_1 = label_data[1,:]
plt.imshow(pic_0)
plt.show()
plt.imshow(pic_1)
plt.show()
a = j
break
'''





