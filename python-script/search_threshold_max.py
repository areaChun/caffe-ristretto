#coding=utf-8
#import libraries
import numpy as np
#import matplotlib.pyplot as plt
import os,sys,caffe,time

matplotlib = 'incline'
model_dir = os.getcwd() +'/'
print 'model_dir:',model_dir
#output_dir = 'models/squeezenet_v1.1/test_nets5/'
#**************************system argument*************************#
deploy_filename = sys.argv[1]
caffemodel_filename = sys.argv[2]

solver_filename = sys.argv[3]
#output_dir = 'models/VGG_ILSVRC_19/test_nets5/'
output_dir = sys.argv[4]
gpu_num = int(sys.argv[5])
#plt.rcParams['figure.figsize'] = (8,8)
#plt.rcParams['image.interpolation'] = 'nearest'
caffe.set_device(gpu_num)
caffe.set_mode_gpu()
solver = None
solver = caffe.get_solver(model_dir+solver_filename)

#caffe.set_mode_cpu()
net = caffe.Net(model_dir+deploy_filename,
	model_dir+caffemodel_filename,
	caffe.TEST)

net_pruned = caffe.Net(model_dir+deploy_filename,
	caffe.TEST)

# net_pro = caffe.Net(model_dir+'trainAlexnetLRN.prototxt',
# 	model_dir+caffemodel_filename,
# 	caffe.TEST)

start = time.time()
# print '********************************************effef'
# print net_pro.forward()['acc']

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
#num_pruned = 0
amount_pruned = 0
amount_weight = 0
#zero_exited = 0
# def weight_prune(layer_original,layer_pruned):
# 	#layer_pruned.reshape(*net.params['conv1'][0].data.shape)
# 	#np.copy(layer_pruned,(layer_original))
# 	layer_pruned.data[...] = layer_original.data
# 	print '**********************'
# 	print layer_pruned.data
# 	num_pruned = 0
# 	zero_exited = 0
# 	#threshold_prune = np.abs(np.mean(np.abs(layer_original.data)))/0.6745
# 	threshold_prune = 0.01
# 	print ('%s: threshold_prune: %.8f ' % (k,threshold_prune))
# 	if (layer_pruned.data.ndim!=1 and layer_pruned.data.ndim!=2):
# 		print 'layer:',k,layer_pruned.data.shape,'dim:',layer_pruned.data.ndim
# 		for kernel in range(layer_pruned.data.shape[0]):
# 		 	for input_num in range(layer_pruned.data.shape[1]):
# 		 		for height in range(layer_pruned.data.shape[2]): 
# 		 			for width in range(layer_pruned.data.shape[3]):
# 		 				#f.write(str(i.data[kernel][input_num][height][width])+' ')
# 		 				buffer_abs = abs(layer_pruned.data[kernel][input_num][height][width])
# 		 				if(buffer_abs == 0):
#  							zero_exited += 1
# 		 				if(buffer_abs<threshold_prune):
# 		 					#print layer_pruned.data[kernel][input_num][height][width],' < ',threshold_prune
# 		 					layer_pruned.data[kernel][input_num][height][width] = 0
# 		 					num_pruned += 1
# 	elif (layer_pruned.data.ndim!=1):
# 		print 'layer:',k,layer_pruned.data.shape,'dim:',layer_pruned.data.ndim
# 		for neturalUnit_num in range(layer_pruned.data.shape[0]):
#  			for map_num in range(layer_pruned.data.shape[1]):
#  				#f.write(str(i.data[neturalUnit_num][map_num])+' ')
#  				buffer_abs = abs(layer_pruned.data[neturalUnit_num][map_num])
#  				if(buffer_abs == 0):
#  					zero_exited += 1
#  				if(buffer_abs<threshold_prune):
#  					layer_pruned.data[neturalUnit_num][map_num] = 0
#  					num_pruned += 1
#  	else:
#  		print 'layer:',k,layer_pruned.data.shape,'dim:',layer_pruned.data.ndim
#  		for neturalUnit_num in range(layer_pruned.data.shape[0]):
#  			buffer_abs = abs(layer_pruned.data[neturalUnit_num])
#  			if(buffer_abs == 0):
#  				zero_exited += 1
#  			if(buffer_abs<threshold_prune):
#  				layer_pruned.data[neturalUnit_num] = 0
#  				num_pruned += 1
#  			#f.write(str(i.data[neturalUnit_num])+' ')
# 		#print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 	#params_dim += 1
# 	print '#######################################'
# 	#print layer_pruned.data
# 	print 'num_pruned: ',num_pruned
# 	print 'zero_exited: ',zero_exited
# 	return num_pruned

# if(os.path.exists('_net.txt')):
# 	os.remove('_net.txt')
# f=open('_net.txt','a')
# for kernel in range(net.params['conv1'][0].data.shape[0]):
# 	for input_num in range(net.params['conv1'][0].data.shape[1]):
# 		print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
# 		for height in range(net.params['conv1'][0].data.shape[2]): 
# 			print >>f,'\n'
# 			for width in range(net.params['conv1'][0].data.shape[3]):
# 				f.write(str(net.params['conv1'][0].data[kernel][input_num][height][width])+' ')


def weight_prune(layer_original,layer_pruned,threshold_prune,k,f):
	#layer_pruned.data[...]=layer_original.data
	#print layer_pruned.data
	zero_exited = 0
	num_pruned = 0
	n = 0
	print 'threshold_prune：',threshold_prune
	#print 'lenv； ', len(layer_original)
	for i in range(len(layer_original)):
		print 'dim: ',i
		layer_pruned[i].data[...] = np.where(abs(layer_original[i].data) >= threshold_prune , layer_original[i].data ,0)
		zero_exited += layer_original[i].data.size-np.count_nonzero(layer_original[i].data)
		num_pruned += layer_original[i].data.size-np.count_nonzero(layer_pruned[i].data)
		n += layer_original[i].data.size
		#print 'dim_pruned: ',num_pruned
	print 'layer_n: ',n
	print 'num_pruned: ',num_pruned
	f.write(str(threshold_prune)+ '\t'+str((num_pruned/float(n))*100.0)+'\n')
	#print >> f,str(threshold_prune)+ '\t'+str((num_pruned/float(layer_original[i].data.size)*100.0))+'\n'
	#print layer_pruned.data
	return num_pruned

def weight_recover(layer_original,layer_pruned,threshold_prune):
	print 'lenv； ', len(layer_original)
	for i in range(len(layer_original)):
		print 'recover-dim: ',i
		layer_pruned[i].data[...] = layer_original[i].data

# start = time.time()
# #solver.net.params['conv1'][0].data[...]=net.params['conv1'][0].data
# #np.copy(solver.net.params['conv1'][0],(net.params['conv1'][0]))
# #print solver.net.params['conv1'][0].data
# amount_pruned += weight_prune(net.params['conv1'],solver.net.params['conv1'],0.1)
# print 'amount-pruned: ',amount_pruned
# print ('\n for %.8f s'%(time.time()-start))
# print 'solver: params:',solver.net.params['conv1'][0].data.shape
# print '***********************************'

def binarySearch(minThreshold,maxThreshold):
	return round((float(minThreshold)+float(maxThreshold))/2.00000,5)

## number of testing  iteration is 100*test_it
def test_accuracy(solver_t):
	accuracy_sum = 0
	for test_it in range(50):
		solver_t.test_nets[0].forward()
		accuracy_sum += solver_t.test_nets[0].blobs['accuracy'].data
		print 'test_accuracy:',solver_t.test_nets[0].blobs['accuracy'].data
		# print 'accuracy: ',solver_t.test_nets[0].blobs['accuracy'].data
		# print 'label argmax: ',solver_t.test_nets[0].blobs['fc8'].data.argmax(1).size
		# print 'sum: ',sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
		# correct += sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
	return accuracy_sum/float(50.0)
def test_accuracy_ending(solver_t):
	accuracy_sum = 0
	for test_it in range(50):
		solver_t.test_nets[0].forward()
		accuracy_sum += solver_t.test_nets[0].blobs['accuracy'].data
		# print 'accuracy: ',solver_t.test_nets[0].blobs['accuracy'].data
		# print 'label argmax: ',solver_t.test_nets[0].blobs['fc8'].data.argmax(1).size
		# print 'sum: ',sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
		# correct += sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
	return accuracy_sum/float(50.0)

def init_testnet(origal_net,solver_net):
	for k,v in origal_net.params.items():
		params_dim = 0
		for i in v :
			#print 'k: ',k,' params_dim: ',params_dim
			solver_net.test_nets[0].params[k][params_dim].data[...] = i.data
			params_dim += 1	

def init_testnet_dist(origal_net,solver_net,bound):
	dist_params = {}
	for k,v in origal_net.params.items():
		z=3
		params_dim = 0
		layer_mean = np.mean(origal_net.params[k][0].data)
		layer_std = np.std(origal_net.params[k][0].data)
		dist_params[k]= [layer_mean,layer_std]
		bound[k] = [0.0,z*layer_std+layer_mean] # according z=(x-mean)/std
		for i in v :
			print 'k: ',k,' params_dim: ',params_dim
			solver_net.test_nets[0].params[k][params_dim].data[...] = i.data
			params_dim += 1
	print 'dist_params: ',dist_params

start = time.time()
layer_bound = {}
layer_threshold = {}
init_testnet_dist(net,solver,layer_bound)
accuray_based=test_accuracy(solver)
print layer_bound
print 'accuray_based: ',accuray_based
print('initial: %.5f \n' % (time.time()-start))
######weight_prune(net.params['conv1'],solver.test_nets[0].params['conv1'],0.15659839)

######print '15659839: ',test_accuracy(solver)

for_time = time.time()
for k,v in reversed(net.params.items()):
	file_dir=os.path.dirname(output_dir +str(k)+'.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir+str(k)+'_prunedRate.txt')):
		os.remove(output_dir+str(k)+'_prunedRate.txt')
	f=open(output_dir+str(k)+'_prunedRate.txt','a')
	print '\n',k,' original: ',solver.test_nets[0].params[k][0].data
	print '\n',k,':'
	bound_bottom = layer_bound[k][0]
	bound_top = layer_bound[k][1]
	bound_median = 0
	pre_median = np.infty
	print ('prune_bottom: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_bottom,k,f))
	accuracy_bottom = test_accuracy(solver)
	print 'accuracy_bottom: ',accuracy_bottom
	weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
	print ('prune_top: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_top,k,f))
	accuracy_top = test_accuracy(solver)
	print 'accuracy_top: ',accuracy_top
	weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
	if(accuracy_bottom < accuray_based - 0.03):
		layer_threshold[k] = layer_bound[k][0]
		continue
	if(accuracy_top >= accuray_based - 0.03):
		layer_threshold[k] = layer_bound[k][1]
		continue
	search_time = time.time()
	same = 0
	n = np.infty
	num = 0
	flag = False
	print '\n\nbound_bottom: ',bound_bottom
	print 'bound_top: ',bound_top
	while same != 1 :
		#weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
		print '\n\nbound_bottom: ',bound_bottom
		print 'bound_top: ',bound_top
		bound_median = binarySearch(bound_bottom,bound_top)
		print 'bound_median: ',bound_median
		num = weight_prune(net.params[k],solver.test_nets[0].params[k],bound_median,k,f)
		print ('prune_median: %d ' % num )
		print 'n: ',n
		accuracy_median = test_accuracy(solver)
		print 'accuracy_median: ',accuracy_median
		if(accuracy_median < accuray_based - 0.03):
			bound_top = bound_median
			flag = False
			print 'bound_top'
		elif(accuracy_median >= accuray_based - 0.03):
			bound_bottom = bound_median
			flag = True
			print 'bound_bottom'
		else:
			flag = True
		if(n == num and flag):
			same = 1
		if(pre_median == bound_median):#extrem predition
			same =1
			bound_median = bound_bottom
		pre_median = bound_median
		n = num
		weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
		#init_testnet(net,solver)
	amount_pruned += num
	print k,': prune_num: ',num
	print k,': threshold: ',bound_median
	layer_threshold[k] = bound_median
	amount_weight += net.params[k][0].data.size
	amount_weight += net.params[k][1].data.size
	print('%s : search_time : %.5f \n' % (k,(time.time()-search_time)))
	print '\n',k,' pruned: ',solver.test_nets[0].params[k][0].data
print 'layer_threshold: ',layer_threshold
print 'ending accuracy: ',test_accuracy(solver)
print 'amount_pruned: ',amount_pruned
print 'amount_weight: ',amount_weight
print 'pruned_rate:',(amount_pruned/float(amount_weight))*100.0
print('for_time : %.5f \n' % (time.time()-for_time))
nsas = os.path.splitext(caffemodel_filename)
if(os.path.exists(nsas[0]+'_pruned.caffemodel')):
 	os.remove(nsas[0]+'_pruned.caffemodel')
solver.test_nets[0].save(nsas[0]+'_pruned.caffemodel')

# for_time = time.time()
# for k,v in net.params.items():
# 	print '\n',k,':'
# 	bound_bottom = layer_bound[k][0]
# 	bound_top = layer_bound[k][1]
# 	bound_median = 0
# 	print ('prune_bottom: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_bottom))
# 	accuracy_bottom = test_accuracy(solver)
# 	print 'accuracy_bottom: ',accuracy_bottom
# 	init_testnet(net,solver)
# 	print ('prune_top: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_top))
# 	accuracy_top = test_accuracy(solver)
# 	print 'accuracy_top: ',accuracy_top
# 	init_testnet(net,solver)
# 	if(accuracy_bottom < accuray_based - 0.03):
# 		layer_threshold[k] = layer_bound[k][0]
# 		continue
# 	if(accuracy_top >= accuray_based - 0.03):
# 		layer_threshold[k] = layer_bound[k][1]
# 		continue
# 	search_time = time.time()
# 	same = 0
# 	n = np.infty
# 	num = 0
# 	flag = False
# 	print '\n\nbound_bottom: ',bound_bottom
# 	print 'bound_top: ',bound_top
# 	while same != 1 :
# 		print '\n\nbound_bottom: ',bound_bottom
# 		print 'bound_top: ',bound_top
# 		bound_median = binarySearch(bound_bottom,bound_top)
# 		print 'bound_median: ',bound_median
# 		num = weight_prune(net.params[k],solver.test_nets[0].params[k],bound_median)
# 		print ('prune_median: %d ' % num )
# 		print 'n: ',n
# 		accuracy_median = test_accuracy(solver)
# 		print 'accuracy_median: ',accuracy_median
# 		if(accuracy_median < accuray_based - 0.03):
# 			bound_top = bound_median
# 			flag = False
# 			print 'bound_top'
# 		elif(accuracy_median >= accuray_based - 0.03):
# 			bound_bottom = bound_median
# 			flag = True
# 			print 'bound_bottom'
# 		else:
# 			flag = True
# 		if(n == num and flag):
# 			same = 1
# 		n = num
# 		init_testnet(net,solver)
# 	print k,': prune_num: ',num
# 	print k,': threshold: ',bound_median
# 	layer_threshold[k] = bound_median
# 	print('%s : search_time : %.5f \n' % (k,(time.time()-search_time)))
# 	break
# print 'layer_threshold: ',layer_threshold
# print('for_time : %.5f \n' % (time.time()-for_time))



# if(os.path.exists('_net.txt')):
# 	os.remove('_net.txt')
# f=open('_net.txt','a')
# for kernel in range(net.params['conv1'][0].data.shape[0]):
# 	for input_num in range(net.params['conv1'][0].data.shape[1]):
# 		print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
# 		for height in range(net.params['conv1'][0].data.shape[2]): 
# 			print >>f,'\n'
# 			for width in range(net.params['conv1'][0].data.shape[3]):
# 				f.write(str(net.params['conv1'][0].data[kernel][input_num][height][width])+' ')
# if(os.path.exists('solver_net.txt')):
# 	os.remove('solver_net.txt')
# f=open('solver_net.txt','a')
# for kernel in range(solver.net.params['conv1'][0].data.shape[0]):
# 	for input_num in range(solver.net.params['conv1'][0].data.shape[1]):
# 		print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
# 		for height in range(solver.net.params['conv1'][0].data.shape[2]): 
# 			print >>f,'\n'
# 			for width in range(solver.net.params['conv1'][0].data.shape[3]):
# 				f.write(str(solver.net.params['conv1'][0].data[kernel][input_num][height][width])+' ')
# if(os.path.exists('buffer_array.txt')):
# 	os.remove('buffer_array.txt')
# f=open('buffer_array.txt','a')
# for kernel in range(buffer_array.shape[0]):
# 	for input_num in range(buffer_array.shape[1]):
# 		print >>f,'\n\n'+'kernel='+str(kernel)+' '+'input_num='+str(input_num)+' '+'\n'
# 		for height in range(buffer_array.shape[2]): 
# 			print >>f,'\n'
# 			for width in range(buffer_array.shape[3]):
# 				f.write(str(buffer_array[kernel][input_num][height][width])+' ')


#*************************************************************
# #threshold = {}
# layer_pruned = 0
# amount_pruned = 0
# zero_exited = 0
# for k,v in net.params.items():
# 	params_dim = 0
# 	buffer_array = np.reshape(net.params[k][0].data,(net.params[k][0].data.size))
# 	#threshold_prune = np.abs(np.std(buffer_array))
# 	threshold_prune = np.abs(np.mean(np.abs(buffer_array)))/0.6745
# 	print ('%s: threshold_prune: %.8f ' % (k,threshold_prune))
# 	for i in v :
# 		if (i.data.ndim!=1 and i.data.ndim!=2):
# 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 			for kernel in range(i.data.shape[0]):
# 		 		for input_num in range(i.data.shape[1]):
# 		 			for height in range(i.data.shape[2]): 
# 		 				for width in range(i.data.shape[3]):
# 		 					#f.write(str(i.data[kernel][input_num][height][width])+' ')
# 		 					buffer_abs = abs(i.data[kernel][input_num][height][width])
# 		 					if(buffer_abs == 0):
#  								zero_exited += 1
# 		 					if(buffer_abs<threshold_prune):
# 		 						i.data[kernel][input_num][height][width] = 0
# 		 						layer_pruned += 1
# 		elif (i.data.ndim!=1):
# 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 			for neturalUnit_num in range(i.data.shape[0]):
#  				for map_num in range(i.data.shape[1]):
#  					#f.write(str(i.data[neturalUnit_num][map_num])+' ')
#  					buffer_abs = abs(i.data[neturalUnit_num][map_num])
#  					if(buffer_abs == 0):
#  						zero_exited += 1
#  					if(buffer_abs<threshold_prune):
#  						i.data[neturalUnit_num][map_num] = 0
#  						layer_pruned += 1
#  		else:
#  			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
#  			for neturalUnit_num in range(i.data.shape[0]):
#  				buffer_abs = abs(i.data[neturalUnit_num])
#  				if(buffer_abs == 0):
#  					zero_exited += 1
#  				if(buffer_abs<threshold_prune):
#  					i.data[neturalUnit_num] = 0
#  					layer_pruned += 1
#  				#f.write(str(i.data[neturalUnit_num])+' ')
# 			#print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 		np.copy(net_pruned.params[k][params_dim],net.params[k][params_dim])
# 		params_dim += 1
# 	print k,':zero = ',zero_exited
# 	print k,':pruned = ',layer_pruned
# 	print ('%s :rate = : %.2f \n' % (k,((layer_pruned/float(layer_param_num[k]))*100)))
# 	amount_pruned += layer_pruned
# 	layer_pruned = 0
# 	zero_exited = 0
# print 'amount_pruned:',amount_pruned
# print ('amount_rate: %.2f ' % ((amount_pruned/float(params_num))*100))

#*************************************************************

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
# for k,v in solver.test_nets[0].params.items():
# 	params_dim = 0
# 	for i in v :
# 		file_dir=os.path.dirname(output_dir +str(k)+'.txt')
# 		if(os.path.exists(file_dir)==False):
# 			os.mkdir(file_dir)
# 			print 'mkdir:',file_dir
# 		if(os.path.exists(output_dir+str(k)+'_param['+str(params_dim)+'].txt')):
# 			os.remove(output_dir+str(k)+'_param['+str(params_dim)+'].txt')
# 		f=open(output_dir+str(k)+'_param['+str(params_dim)+'].txt','a')
# 		params_dim += 1
# 		if (i.data.ndim!=1 and i.data.ndim!=2):
# 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 			for kernel in range(i.data.shape[0]):
# 		 		for input_num in range(i.data.shape[1]):
# 		 			#print >>f,' '+'kernel='+str(kernel)+' '+'input_num='+str(input_num)
# 		 			for height in range(i.data.shape[2]): 
# 		 				#print >>f
# 		 				for width in range(i.data.shape[3]):
# 		 					f.write(str(i.data[kernel][input_num][height][width])+' ')
# 		elif (i.data.ndim!=1):
# 			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
# 			for neturalUnit_num in range(i.data.shape[0]):
# 				#print >>f,'\n\n neturalUnit_num='+str(neturalUnit_num)+'\n'
#  				for map_num in range(i.data.shape[1]):
#  					f.write(str(i.data[neturalUnit_num][map_num])+' ')
#  		else:
#  			print 'layer:',k,i.data.shape,'dim:',i.data.ndim
#  			for neturalUnit_num in range(i.data.shape[0]):
#  					f.write(str(i.data[neturalUnit_num])+' ')
#*********************************************************#

# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
# nsas = os.path.splitext(caffemodel_filename)
# if(os.path.exists(nsas[0]+'_pruned.caffemodel')):
#  	os.remove(nsas[0]+'_pruned.caffemodel')
# net.save(nsas[0]+'_pruned.caffemodel')


# print("\nDone in %.2f s.\n" % (time.time() - start))#计算时间
