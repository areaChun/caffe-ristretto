#coding=utf-8
#import libraries
import numpy as np
#import matplotlib.pyplot as plt
import os,sys,caffe,time

matplotlib = 'incline'
model_dir = os.getcwd() +'/'
# deploy_filename = 'deploy.prototxt'
# caffemodel_filename = 'bvlc_googlenet.caffemodel'
# img_root = 'cat.jpg'
# bin_mean_filename = 'imagenet_mean.binaryproto'
# npy_mean_filename = 'imagenet_mean.npy'
# labels_filename = 'synset_words.txt'

#**************************system argument*************************#
deploy_filename = sys.argv[1]
caffemodel_filename = sys.argv[2]
img_root = sys.argv[3]
bin_mean_filename = sys.argv[4]
npy_mean_filename = 'imagenet_mean.npy'
labels_filename = sys.argv[5]
output_dir = 'output/'
#plt.rcParams['figure.figsize'] = (8,8)
#plt.rcParams['image.interpolation'] = 'nearest'

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(2)
net = caffe.Net(model_dir+deploy_filename,
	model_dir+caffemodel_filename,
	caffe.TEST)

#**************************show_data function*************************#
# def show_data(data,title,padsize=1,padval=0):
# 	data -= data.min()
# 	data /= data.max()
# 	n = int(np.ceil(np.sqrt(data.shape[0])))
# 	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
# 	data = np.pad(data,padding,mode='constant',constant_values=(padval, padval))
# 	# tile the filters into an image
# 	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
# 	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
# 	plt.figure(), plt.title(title)
# 	plt.imshow(data, cmap='gray')
# 	# plt.imshow(data)
# 	plt.axis('off')

def convert_mean(binMean, npyMean):
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open(binMean, 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save(npyMean, npy_mean)
binMean = model_dir + bin_mean_filename
npyMean = model_dir + npy_mean_filename
convert_mean(binMean, npyMean)

def statistics(dict_z):
	dict_ori = {}
	zero_exited_net = 0
	n = 0
	#print 'threshold_prune：',threshold_prune
	#print 'lenv； ', len(layer_original)
	for k,v in net.params.items():
		zero_exited_layer = 0
		num_pruned_layer = 0
		m = 0
		for i in v:
			zero_exited_layer += i.data.size-np.count_nonzero(i.data)	
			n += i.data.size
			m += i.data.size
		dict_z[k] = zero_exited_layer
		dict_ori[k] = m
		zero_exited_net += zero_exited_layer
	dict_z['net'] = zero_exited_net
	dict_ori['net'] = n
	return dict_ori
#start = time.time()

im = caffe.io.load_image(model_dir + img_root)



transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].data[...] = transformer.preprocess('data', im)



out = net.forward()

dict_pre = {}
dict_after = {}
dict_net = statistics(dict_pre)
	#params_output_first()
	#train(1000,20)
statistics(dict_after)
for k,v in net.params.items():
	print '\n',k,': '
	print 'pre: ',dict_pre[k],' rate: ',(dict_pre[k]/float(dict_net[k])*100.0)
	#print 'aft: ',(dict_after[k]/float(dict_net[k])*100.0)
	#print 'up :',(dict_after[k]/float(dict_net[k])*100.0) - (dict_pre[k]/float(dict_net[k])*100.0)
print '\n','net: '
print 'pre: ',dict_pre['net'],' rate: ',(dict_pre['net']/float(dict_net['net'])*100.0)
#print 'aft: ',(dict_after['net']/float(dict_net['net'])*100.0)
#print 'up :',(dict_after['net']/float(dict_net['net'])*100.0) - (dict_pre['net']/float(dict_net['net'])*100.0)
#print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间




# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
