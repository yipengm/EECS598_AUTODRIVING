from PIL import Image

f = open("TEST_name")

for line in f:
	rgb_name = line[0:-1]+"_image.jpg"
	dep_name = line[0:-1]+"_depth.jpg"

	image_rgb = Image.open(rgb_name)
	image_dep = Image.open(dep_name)
	new_rgb = image_rgb.resize((256,256))
	new_dep = image_dep.resize((256,256))
	new_rgb.save(line[0:-1]+"_image_256.jpg")
	new_dep.save(line[0:-1]+"_depth_256.jpg")
	print(rgb_name)
