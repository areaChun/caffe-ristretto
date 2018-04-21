#coding=utf-8
#import libraries
import numpy as np
#import matplotlib.pyplot as plt
import os,sys,caffe,time

matplotlib = 'incline'
model_dir = os.getcwd() +'/'

#**************************system argument*************************#
deploy_filename = sys.argv[1]
caffemodel_filename = sys.argv[2]
gpu_num = int(sys.argv[3])

caffe.set_device(gpu_num)
caffe.set_mode_gpu()
#plt.rcParams['figure.figsize'] = (8,8)
#plt.rcParams['image.interpolation'] = 'nearest'


#caffe.set_mode_cpu()
net = caffe.Net(model_dir+deploy_filename,
	model_dir+caffemodel_filename,
	caffe.TEST)

net_pruned = caffe.Net(model_dir+deploy_filename,
	caffe.TEST)


start = time.time()


#*********************** pause **********************************#
#name = input("Please input your name:\n")

conv_bias_amount = 0
conv_param_amount = 0
BN_param_amount = 0
fc_param_amount = 0
fc_bias_amount = 0 

visualizationLayer_activation_amount = 0
fc_activation_amount = 0
params_num = 0
layer_param_num = {}
for k,v in net.params.items():
	#***********************Blobvec dim **********************************#
	n=0
	v_dim = len(v)
	#print 'v: ',len(v)
	#*********************************************************************#
	fc_status = 0
	for i in v :
		if (i.data.ndim!=1 and i.data.ndim!=2):
			conv_param_amount += i.data.shape[0]*i.data.shape[1]*i.data.shape[2]*i.data.shape[3]
			n += i.data.shape[0]*i.data.shape[1]*i.data.shape[2]*i.data.shape[3]
			print str(k),' conv_param_amount :',conv_param_amount,'(',i.data.shape[0]*i.data.shape[1]*i.data.shape[2]*i.data.shape[3],')'
		elif (i.data.ndim!=1):
			fc_status += 1
			fc_param_amount += i.data.shape[0]*i.data.shape[1]
			n += i.data.shape[0]*i.data.shape[1]
			print str(k),' fc_param_amount :',fc_param_amount,'(',i.data.shape[0]*i.data.shape[1],')'
 		else:
 			if (v_dim == 3):
 				BN_param_amount += i.data.shape[0]
 				n += i.data.shape[0]
 				print str(k),' BN_param_amount :',BN_param_amount,'(',i.data.shape[0],')'
			else:
				if (fc_status == 0):
					conv_bias_amount += i.data.shape[0]
					n += i.data.shape[0]
					print str(k),' conv_bias_amount :',conv_bias_amount,'(',i.data.shape[0],')'
				else:
					fc_bias_amount += i.data.shape[0]
					n += i.data.shape[0]
					print str(k),' fc_bias_amount :',fc_bias_amount,'(',i.data.shape[0],')'
					fc_status = 0
	layer_param_num[k] = n
	params_num += n
print layer_param_num
print 'visualizationLayer_activation_amount:',visualizationLayer_activation_amount
print 'fc_activation_amount:',fc_activation_amount
print 'conv_param_amount:',conv_param_amount
print 'conv_bias_amount:',conv_bias_amount
print 'BN_param_amount:',BN_param_amount
print 'fc_param_amount:',fc_param_amount
print 'fc_bias_amount:',fc_bias_amount
print 'all_params_number:',params_num



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
#threshold_dict = {'conv1':0.05857147,'conv2':0.02208223,'conv3':0.0,'conv4':0.0,'conv5':0.01595983,'fc6':0.0,'fc7': 0.00384274,'fc8':0.00963896,}
threshold_dict = {'conv1':0.0,'conv2':0.0,'conv3':0.0,'conv4':0.0,'conv5':0.0,'fc6':0.0,'fc7': 0.0,'fc8':0.0107}
#threshold = {}
layer_pruned = 0
amount_pruned = 0
zero_exited = 0
for k,v in net.params.items():
	params_dim = 0
	buffer_array = np.reshape(net.params[k][0].data,(net.params[k][0].data.size))
	#threshold_prune = np.abs(np.std(buffer_array))
	#threshold_prune = np.abs(np.mean(np.abs(buffer_array)))/0.6745
	threshold_prune = threshold_dict[k]
	print ('%s: threshold_prune: %.8f ' % (k,threshold_prune))
	for i in v :
		if (i.data.ndim!=1 and i.data.ndim!=2):
			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
			for kernel in range(i.data.shape[0]):
		 		for input_num in range(i.data.shape[1]):
		 			for height in range(i.data.shape[2]): 
		 				for width in range(i.data.shape[3]):
		 					#f.write(str(i.data[kernel][input_num][height][width])+' ')
		 					buffer_abs = abs(i.data[kernel][input_num][height][width])
		 					if(buffer_abs == 0):
 								zero_exited += 1
		 					if(buffer_abs<threshold_prune):
		 						i.data[kernel][input_num][height][width] = 0
		 						layer_pruned += 1
		elif (i.data.ndim!=1):
			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
			for neturalUnit_num in range(i.data.shape[0]):
 				for map_num in range(i.data.shape[1]):
 					#f.write(str(i.data[neturalUnit_num][map_num])+' ')
 					buffer_abs = abs(i.data[neturalUnit_num][map_num])
 					if(buffer_abs == 0):
 						zero_exited += 1
 					if(buffer_abs<threshold_prune):
 						i.data[neturalUnit_num][map_num] = 0
 						layer_pruned += 1
 		else:
 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
 			for neturalUnit_num in range(i.data.shape[0]):
 				buffer_abs = abs(i.data[neturalUnit_num])
 				if(buffer_abs == 0):
 					zero_exited += 1
 				if(buffer_abs<threshold_prune):
 					i.data[neturalUnit_num] = 0
 					layer_pruned += 1
 				#f.write(str(i.data[neturalUnit_num])+' ')
			#print 'layer:',k,i.data.shape,'dim:',i.data.ndim
		np.copy(net_pruned.params[k][params_dim],net.params[k][params_dim])
		params_dim += 1
	print k,':zero = ',zero_exited
	print k,':pruned = ',layer_pruned
	print ('%s :rate = : %.2f \n' % (k,((layer_pruned/float(layer_param_num[k]))*100)))
	amount_pruned += layer_pruned
	layer_pruned = 0
	zero_exited = 0
print 'amount_pruned:',amount_pruned
print ('amount_rate: %.2f ' % ((amount_pruned/float(params_num))*100))

# for k,v in net.params.items():
# 	params_dim = 0
# 	for i in v :
# 		np.copy(net_pruned.params[k][params_dim],net.params[k][params_dim])
# 		params_dim += 1
		# if (i.data.ndim!=1 and i.data.ndim!=2):
		# 	print 'layer:',k,i.data.shape,'dim:',i.data.ndim
		# 	for kernel in range(i.data.shape[0]):
		#  		for input_num in range(i.data.shape[1]):
		#  			print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
		#  			for height in range(i.data.shape[2]): 
		#  				print >>f,'\n'
		#  				for width in range(i.data.shape[3]):
		#  					#f.write(str(i.data[kernel][input_num][height][width])+' ')
		#  					i.data[kernel][input_num][height][width] = 2.0
		# elif (i.data.ndim!=1):
		# 	print 'layer:',k,i.data.shape,'dim:',i.data.ndim
		# 	for neturalUnit_num in range(i.data.shape[0]):
		# 		print >>f,'\n\n neturalUnit_num='+str(neturalUnit_num)+'\n'
 	# 			for map_num in range(i.data.shape[1]):
 	# 				#f.write(str(i.data[neturalUnit_num][map_num])+' ')
 	# 				i.data[neturalUnit_num][map_num] = 3.0
 	# 	else:
 	# 		print 'layer:',k,i.data.shape,'dim:',i.data.ndim
 	# 		for neturalUnit_num in range(i.data.shape[0]):
 	# 			i.data[neturalUnit_num] = 4.0
 				#f.write(str(i.data[neturalUnit_num])+' ')
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
nsas = os.path.splitext(caffemodel_filename)
if(os.path.exists(nsas[0]+'_pruned.caffemodel')):
 	os.remove(nsas[0]+'_pruned.caffemodel')
net.save(nsas[0]+'_pruned.caffemodel')


# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
