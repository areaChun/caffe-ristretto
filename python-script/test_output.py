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
caffe.set_device(1)
net = caffe.Net(model_dir+deploy_filename,
	model_dir+caffemodel_filename,
	caffe.TEST)
for name,layer in zip(net._layer_names,net.layers):
	print "%s  : %s : %d blobs"%(name,layer.type,len(layer.blobs))
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

#start = time.time()

im = caffe.io.load_image(model_dir + img_root)



transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].data[...] = transformer.preprocess('data', im)



out = net.forward()

#**************************a test*************************#
#print '\ndata_shape\n'
#print [(k, v.data.shape) for k, v in net.blobs.items()]
#print '\nbias_shape\n'
##print [(k, v[1].data.shape) for k, v in net.params.items()]
#print '\nparams_shape\n'
#print [(k, v[0].data.shape) for k, v in net.params.items()]
# start = time.time()
# for k,v in net.blobs.items():
# 	print 'layer:',k,v.data.shape
# 	if(os.path.exists(output_dir+'vgg19_'+str(k)+'.txt')):
# 		os.remove(output_dir+'vgg19_'+str(k)+'.txt')
# 	f=open(output_dir+'vgg19__'+str(k)+'.txt','a')
# 	if(v.data.ndim!=2):
# 		#print 'shape:',v.data.shape[0],v.data.shape[1],v.data.shape[2],v.data.shape[3]
# 		# print 'conv1_1:',net.blobs['conv1_1'].data.shape
# 		f.write('\nconv1_1 layer:'+' \n')
# 		for batch in v.data:
# 	  		for output_num in batch:
# 	  			#print >>f,'\n\n'+'batch='+str(batch)+' '+'output_num='+str(output_num)+' '
# 	  			for height in output_num:
# 	  		 		#print >>f,'\n'
# 	  		 		for width in height:
# 	  					#f.write(str(width)+' ')
# 	  					nnan=1
# 	else:
# 		for batch in v.data:
# 		  	for output_num in batch:
# 		  		#.write(str(output_num)+' ')
# 		  		nnan=1
# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
#print '\ndata_shape\n'
#print [(k, v.data.shape) for k, v in net.blobs.items()]
#print '\nbias_shape\n'
# print [(k, v[1].data.shape) for k, v in net.params.items()]
#print '\nparams_shape\n'
#print [(k, v[0].data.shape) for k, v in net.params.items()]
#*********************************************************#


start = time.time()
print '\ndata_shape\n'
file_dir=os.path.dirname(output_dir +'data_shape.txt')
if(os.path.exists(file_dir)==False):
	os.mkdir(file_dir)
	print 'mkdir:',file_dir
if(os.path.exists(output_dir+'data_shape'+'.txt')):
	os.remove(output_dir+'data_shape'+'.txt')
f=open(output_dir+'data_shape'+'.txt','a')
for k, v in net.blobs.items():
	print >>f,k,v.data.shape

print '\nparams_shape\n'
if(os.path.exists(output_dir+'params_shape'+'.txt')):
	os.remove(output_dir+'params_shape'+'.txt')
f=open(output_dir+'params_shape'+'.txt','a')
for k,v in net.params.items():
	params_dim = 0
	for i in v :
		print >>f,k,'['+str(params_dim)+']',i.data.shape
		params_dim +=1

#*********************** pause **********************************#
#name = input("Please input your name:\n")

conv_bias_amount = 0
conv_param_amount = 0
BN_param_amount = 0
fc_param_amount = 0
fc_bias_amount = 0 

visualizationLayer_activation_amount = 0
fc_activation_amount = 0

for k,v in net.blobs.items():
	if(v.data.ndim!=2):
		#print 'shape:',v.data.shape[0],v.data.shape[1],v.data.shape[2],v.data.shape[3]
		# print 'conv1_1:',net.blobs['conv1_1'].data.shape
		print str(k)
		visualizationLayer_activation_amount += v.data.shape[0]*v.data.shape[1]*v.data.shape[2]*v.data.shape[3]
		print str(k),' visualizationLayer_activation_amount :',visualizationLayer_activation_amount,'(',v.data.shape[0]*v.data.shape[1]*v.data.shape[2]*v.data.shape[3],')'
	else:
		fc_activation_amount += v.data.shape[0]* v.data.shape[1]
		print str(k),' fc_activation_amount :',fc_activation_amount,'(',v.data.shape[0]* v.data.shape[1],')'
v_dim = 0
for k,v in net.params.items():
	#***********************Blobvec dim **********************************#
	v_dim = len(v)
	#print 'v: ',len(v)
	#*********************************************************************#
	fc_status = 0
	for i in v :
		if (i.data.ndim!=1 and i.data.ndim!=2):
			conv_param_amount += i.data.shape[0]*i.data.shape[1]*i.data.shape[2]*i.data.shape[3]
			print str(k),' conv_param_amount :',conv_param_amount,'(',i.data.shape[0]*i.data.shape[1]*i.data.shape[2]*i.data.shape[3],')'
		elif (i.data.ndim!=1):
			fc_status += 1
			fc_param_amount += i.data.shape[0]*i.data.shape[1]
			print str(k),' fc_param_amount :',fc_param_amount,'(',i.data.shape[0]*i.data.shape[1],')'
 		else:
 			if (v_dim == 3):
 				BN_param_amount += i.data.shape[0]
 				print str(k),' BN_param_amount :',BN_param_amount,'(',i.data.shape[0],')'
			else:
				if (fc_status == 0):
					conv_bias_amount += i.data.shape[0]
					print str(k),' conv_bias_amount :',conv_bias_amount,'(',i.data.shape[0],')'
				else:
					fc_bias_amount += i.data.shape[0]
					print str(k),' fc_bias_amount :',fc_bias_amount,'(',i.data.shape[0],')'
					fc_status = 0

print 'visualizationLayer_activation_amount:',visualizationLayer_activation_amount
print 'fc_activation_amount:',fc_activation_amount
print 'conv_param_amount:',conv_param_amount
print 'conv_bias_amount:',conv_bias_amount
print 'BN_param_amount:',BN_param_amount
print 'fc_param_amount:',fc_param_amount
print 'fc_bias_amount:',fc_bias_amount





###**********************output data*****************************###
for k,v in net.blobs.items():
	print 'layer:',k,v.data.shape
	file_dir=os.path.dirname(output_dir +str(k)+'.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir+'vgg4*'+str(k)+'.txt')):
		os.remove(output_dir+str(k)+'.txt')
	f=open(output_dir+str(k)+'.txt','a')
	if(v.data.ndim!=2):
		#print 'shape:',v.data.shape[0],v.data.shape[1],v.data.shape[2],v.data.shape[3]
		# print 'conv1_1:',net.blobs['conv1_1'].data.shape
		# f.write('\nconv1_1 layer:'+' \n')
		for batch in range(v.data.shape[0]):
		  	for output_num in range(v.data.shape[1]):
		  		# print >>f,'\n\n'+'batch='+str(batch)+' '+'output_num='+str(output_num)+' '
		  		for height in range(v.data.shape[2]):
		  		 	# print >>f,'\n'
		  		 	for width in range(v.data.shape[3]): 
		  				f.write(str(v.data[batch][output_num][height][width])+' ')
	else:
		for batch in range(v.data.shape[0]):
		  	for output_num in range(v.data.shape[1]):
		  		f.write(str(v.data[batch][output_num])+' ')
				# #print 'shape:',v.data.shape[0],v.data.shape[1]
				# #print 'shape:',v.data.shape[0],v.data.shape[1]


# ###**********************v2 writting  bias*****************************###
#for k,v in net.params.items():
#	#print 'layer:',k
#	#print 'shape:',v[1].data.shape[0]
#	print 'bias:',k,v[1].data.shape
#	if(os.path.exists(output_dir+'vgg19_'+str(k)+'_bias.txt')):
#		os.remove(output_dir+'vgg19_'+str(k)+'_bias.txt')
#	f=open(output_dir+'vgg19_'+str(k)+'_bias.txt','a')
#	for kernel in range(v[1].data.shape[0]):
#		f.write(str(net.params[k][1].data[kernel])+'  ')
#*********************************************************#

for k,v in net.params.items():
	params_dim = 0
	for i in v :
		file_dir=os.path.dirname(output_dir +str(k)+'.txt')
		if(os.path.exists(file_dir)==False):
			os.mkdir(file_dir)
			print 'mkdir:',file_dir
		if(os.path.exists(output_dir+str(k)+'_param['+str(params_dim)+'].txt')):
			os.remove(output_dir+str(k)+'_param['+str(params_dim)+'].txt')
		f=open(output_dir+str(k)+'_param['+str(params_dim)+'].txt','a')
		params_dim += 1
		if (i.data.ndim!=1 and i.data.ndim!=2):
			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
			for kernel in range(i.data.shape[0]):
		 		for input_num in range(i.data.shape[1]):
		 			#print >>f,' '+'kernel='+str(kernel)+' '+'input_num='+str(input_num)
		 			for height in range(i.data.shape[2]): 
		 				#print >>f
		 				for width in range(i.data.shape[3]):
		 					f.write(str(i.data[kernel][input_num][height][width])+' ')
		elif (i.data.ndim!=1):
			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
			for neturalUnit_num in range(i.data.shape[0]):
				#print >>f,'\n\n neturalUnit_num='+str(neturalUnit_num)+'\n'
 				for map_num in range(i.data.shape[1]):
 					f.write(str(i.data[neturalUnit_num][map_num])+' ')
 		else:
 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
 			for neturalUnit_num in range(i.data.shape[0]):
 					f.write(str(i.data[neturalUnit_num])+' ')
			#print 'layer:',k,i.data.shape,'dim:',i.data.ndim



# ##*****************************************v2 writing the params*************************##
# for k,v in net.params.items():
# 	print 'params:',k,v[0].data.shape
# 	if(os.path.exists(output_dir+'vgg19_'+str(k)+'_param.txt')):
# 		os.remove(output_dir+'vgg19_'+str(k)+'_param.txt')
# 	f=open(output_dir+'vgg19_'+str(k)+'_param.txt','a')
# 	if(v[0].data.ndim!=2):
# 		for kernel in range(v[0].data.shape[0]):
# 		 	for input_num in range(v[0].data.shape[1]):
# 		 		#print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
# 		 		for height in range(v[0].data.shape[2]): 
# 		 			#print >>f,'\n'
# 		 			for width in range(v[0].data.shape[3]):
# 		 				f.write(str(net.params[k][0].data[kernel][input_num][height][width])+' ')
# 		#print 'shape:',v[0].data.shape[0],v[0].data.shape[1],v[0].data.shape[2],v[0].data.shape[3]
# 	else:
# 		for neturalUnit_num in range(v[0].data.shape[0]):
# 			#print >>f,'\n\n neturalUnit_num='+str(neturalUnit_num)+'\n'
#  			for map_num in range(v[0].data.shape[1]):
#  				f.write(str(net.params[k][0].data[neturalUnit_num][map_num])+' ')
# 		#print 'shape:',v[0].data.shape[0],v[0].data.shape[1]
#*********************************************************#

print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间




# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
