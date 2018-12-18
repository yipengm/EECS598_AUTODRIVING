import tensorflow as tf
from SHnet import stacked_hourglass as SHnet
from last_loc_wh import last
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
tf.set_random_seed(0)
fo = open('task1_final.csv','w')
def rot(n):
	n = np.asarray(n).flatten()
	assert(n.size == 3)

	theta = np.linalg.norm(n)
	if theta:
		n /= theta
		K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
		return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
	else:
		return np.identity(3)


def get_bbox(p0, p1):
	v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
	e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)
	return v, e

batch_size = 1
image_size = 256
regularize_para = 0.4

img_rgb_in = tf.placeholder(tf.float32, [None, image_size*image_size*3])

img_depth_in = tf.placeholder(tf.float32, [None, image_size*image_size])

img_rgb = tf.reshape(img_rgb_in,[-1, image_size, image_size, 3])

img_depth = tf.reshape(img_depth_in,[-1,image_size,image_size,1])

inputs = tf.concat([img_rgb,img_depth],3)

center_in = tf.placeholder(tf.float32,[None,3])

GT_wh = tf.placeholder(tf.float32,[None,2])

label_in = tf.placeholder(tf.float32,[None,2])

img2dark = tf.layers.conv2d(inputs = inputs, filters = 256,kernel_size = 1, padding = 'same')
with tf.name_scope("SH2"):
	dark_out1 = SHnet(img2dark)

with tf.name_scope("SH3"):
	dark_out = SHnet(dark_out1)

#center_loss1,class_loss1,_ = last(dark_out1,center_in,label_in)
loss_ROI,pred_center,pred_wh,loss_center,loss_wh = last(dark_out,center_in,GT_wh)
row_center = pred_center[:,1]
col_center = pred_center[:,0]
thres = tf.ones(batch_size)*7
height = tf.maximum(pred_wh[:,0],thres)
width = tf.maximum(pred_wh[:,1],thres) #check the code you will see h => 0 w => 1

Var2restore = tf.global_variables()

with tf.name_scope("cls"):
	bbox2crop0 = tf.reshape((row_center-1/2*height)/256,[-1,1])
	bbox2crop1 = tf.reshape((col_center-1/2*width)/256,[-1,1])
	bbox2crop2 = tf.reshape((row_center+1/2*height)/256,[-1,1])
	bbox2crop3 = tf.reshape((col_center+1/2*width)/256,[-1,1])

	bbox2crop = tf.concat([bbox2crop0,bbox2crop1,bbox2crop2,bbox2crop3],1)
	box_ind = tf.constant([0],tf.int32)
	crop_size = tf.constant([7,7],tf.int32)
	feature2cls = tf.image.crop_and_resize(dark_out,bbox2crop,box_ind,crop_size)
	afterfc = tf.layers.conv2d(inputs = feature2cls, filters = 1024,kernel_size = 1, padding = 'same')
	after_p = tf.layers.average_pooling2d(afterfc,[7,7],1)
	squeeze = tf.reshape(after_p,[-1,1024])
	out1 = tf.layers.dense(squeeze,256,activation=tf.nn.elu)
	out = tf.layers.dense(squeeze,2)
	pred_cls = tf.arg_max(out+tf.constant([1.0,0.0]),-1)
	out_after_bn = tf.layers.batch_normalization(inputs = out,training = 1,axis = -1)
	class_poss = tf.nn.softmax(out_after_bn,-1)
	class_loss = tf.reduce_mean(-tf.reduce_sum(label_in*tf.log(class_poss),1))
	total_loss = class_loss+loss_ROI

Var2train = []

for i in tf.global_variables():
	if i not in Var2restore:
		Var2train.append(i)

initial_learning_rate = 5e-5

train_op = tf.train.AdamOptimizer(learning_rate = initial_learning_rate).minimize(total_loss,var_list = Var2train)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver1 = tf.train.Saver(Var2restore)
saver2 = tf.train.Saver(tf.global_variables())
writer = tf.summary.FileWriter("logs/loc",  tf.get_default_graph())
saver2.restore(sess,'./ckpt_all/cur.ckpt')

im_name_array = []
with open('TEST') as my_file:
	for line in my_file:
		im_name_array.append(line)
epochs = 1
data_num = len(im_name_array)

correct = 0
wrong = 0

for epoch in range(epochs):
	index_rand = range(data_num)
	for j in range(int(data_num/batch_size)):
		image_data = np.zeros([batch_size,image_size,image_size,3])
		depth_data = np.zeros([batch_size,image_size,image_size])
		label_data = np.zeros([batch_size,2])
		label_check = np.zeros([batch_size,1])
		centre_data = np.zeros([batch_size,3])
		WH_data = np.zeros([batch_size,2])
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
			#bbox_path = name_base+"_bbox.bin"
			#centre = np.fromfile(bbox_path, dtype=np.float32)

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
			#centre_vec = np.zeros([4],np.float32)
			#centre_vec[0:3] = centre[3:6]
			#centre_vec[3] = 1
			proj = np.fromfile(name_base+'_proj.bin', dtype=np.float32)
			proj.resize([3, 4])
			#center_projected = proj @ centre_vec
			#centre_data[i, 0] = center_projected[0]/center_projected[2]/1914*256
			#centre_data[i, 1] = center_projected[1]/center_projected[2]/1052*256
			#centre_data[i, 2] = center_projected[2]
			#centre = centre.reshape([-1, 11])
			'''
			for k, b in enumerate(centre):
				R = rot(b[0:3])
				t = b[3:6]

				sz = b[6:9]
				vert_3D, edges = get_bbox(-sz / 2, sz / 2)
				vert_3D = R @ vert_3D + t[:, np.newaxis]

				vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
				vert_2D = vert_2D / vert_2D[2, :]

				min_cor_x = min(vert_2D[0,:])
				max_cor_x = max(vert_2D[0,:])
				min_cor_y = min(vert_2D[1,:])
				max_cor_y = max(vert_2D[1,:])
				Height_2d = max_cor_y - min_cor_y
				width_2d = max_cor_x - min_cor_x
			WH_data[i,0] = Height_2d/1052*256
			WH_data[i,1] = width_2d/1914*256
			'''

		depth_data_in = np.reshape(depth_data,[batch_size,256*256])/255
		image_data_in = np.reshape(image_data,[batch_size,256*256*3])/255
		pred_cls_ = sess.run([pred_cls],feed_dict={img_rgb_in: image_data_in, img_depth_in: depth_data_in})

		print(im_name_array[index_rand[index]][0:-3]+','+str(2-pred_cls_[0][0]),file = fo)
		print(j)

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





