#coding=utf-8
#import libraries
import numpy as np
#import matplotlib.pyplot as plt
import os,sys,caffe,time,math

matplotlib = 'incline'
model_dir = os.getcwd() +'/'
print 'model_dir:',model_dir

#**************************system argument*************************#
solver_name = sys.argv[1]
caffemodel_pruned_name = sys.argv[2]

#solver_filename = sys.argv[3]
#output_dir = 'models/VGG_ILSVRC_19/test_nets5/'
output_dir = sys.argv[4]
gpu_num = int(sys.argv[3])
#plt.rcParams['figure.figsize'] = (8,8)
#plt.rcParams['image.interpolation'] = 'nearest'
caffe.set_device(gpu_num)
caffe.set_mode_gpu()
solver = None
solver = caffe.get_solver(solver_name)
solver.net.copy_from(caffemodel_pruned_name)
#caffe.set_mode_cpu()
# net = caffe.Net(model_dir+deploy_filename,
# 	model_dir+caffemodel_filename,
# 	caffe.TEST)

# net_pruned = caffe.Net(model_dir+deploy_filename,
# 	caffe.TEST)

# net_pro = caffe.Net(model_dir+'trainAlexnetLRN.prototxt',
# 	model_dir+caffemodel_filename,
# 	caffe.TEST)

start = time.time()

#threshold = {'fc6': 0.00447, 'fc7': 0.00972, 'fc8': 0.03114, 'conv3': 0.01228, 'conv2': 0.01324, 'conv1': 0.04432, 'conv5': 0.00829, 'conv4': 0.01148}
#cifar10
#threshold = {'conv3': 0.04065, 'conv2': 0.03701, 'conv1': 0.0459, 'ip1': 0.010605613189889115}
#caffenet
#threshold = {'fc6': 0.00811, 'fc7': 0.00889, 'fc8': 0.01007, 'conv3': 0.00941, 'conv2': 0.01183, 'conv1': 0.012, 'conv5': 0.01756, 'conv4': 0.01321}
#squeezenet
#threshold = {'fire5/squeeze1x1': 0.0227, 'fire2/expand1x1': 0.0, 'fire8/squeeze1x1': 0.01726, \
#				'fire3/expand3x3': 0.00829, 'fire7/expand1x1': 0.01866, 'fire5/expand3x3': 0.02165, \
#				'fire4/expand1x1': 0.04631, 'fire6/expand1x1': 0.04654, 'fire6/expand3x3': 0.00814, \
#				'fire3/expand1x1': 0.07184, 'fire9/expand1x1': 0.0359, 'conv1': 0.04828, \
#				'fire6/squeeze1x1': 0.01102, 'fire7/squeeze1x1': 0.03029, \
#				'fire9/expand3x3': 0.00422, 'fire8/expand1x1': 0.00587, 'conv10': 0.02166, \
#				'fire2/expand3x3': 0.01355, 'fire2/squeeze1x1': 0.06767, 'fire4/squeeze1x1': 0.01633, \
#				'fire7/expand3x3': 0.00787, 'fire5/expand1x1': 0.01766, 'fire3/squeeze1x1': 0.0, \
#				'fire8/expand3x3': 0.0224, 'fire9/squeeze1x1': 0.01243, 'fire4/expand3x3': 0.01819}
# threshold = {'fire2/expand1x1': 0.0, 'fire2/expand3x3': 0.02355, 'fire2/squeeze1x1': 0.01567, \
# 				'fire3/expand1x1': 0.01584, 'fire3/expand3x3': 0.02329, 'fire3/squeeze1x1': 0.0, \
# 				'fire4/expand1x1': 0.02031, 'fire4/expand3x3': 0.03019, 'fire4/squeeze1x1': 0.01633, \
# 				'fire5/expand1x1': 0.01766, 'fire5/expand3x3': 0.03065, 'fire5/squeeze1x1': 0.0207, \
# 				'fire6/expand1x1': 0.02054, 'fire6/expand3x3': 0.02514, 'fire6/squeeze1x1': 0.01102,\
# 				'fire7/expand1x1': 0.01866, 'fire7/expand3x3': 0.02287, 'fire7/squeeze1x1': 0.01829, \
# 				'fire8/expand1x1': 0.00587, 'fire8/expand3x3': 0.0324, 'fire8/squeeze1x1': 0.01726, \
# 				'fire9/expand1x1': 0.0359, 'fire9/expand3x3': 0.03122, 'fire9/squeeze1x1': 0.00843, \
# 				'conv1': 0.04828, 'conv10': 0.01866}
#vgg16
threshold = {'conv1_1': 0.06092,'conv1_2': 0.01003,\
			'conv2_1': 0.01074, 'conv2_2': 0.00752,\
			'conv3_1': 0.00654,'conv3_2': 0.00454,'conv3_3': 0.00393,\
			'conv4_1': 0.00482,'conv4_2': 0.00439, 'conv4_3': 0.00452,\
			'conv5_1': 0.00429,'conv5_2': 0.0045,'conv5_3': 0.00458,  \
			'fc6': 0.00355,'fc7': 0.00603,'fc8': 0.00748}
#num_pruned = 0
amount_pruned = 0
amount_weight = 0
dict_pre = {}
dict_after = {}
dict_net = {}

def weight_prune(solver_pruned,threshold_prune):
	#layer_pruned.data[...]=layer_original.data
	#print layer_pruned.data
	zero_exited_net = 0
	num_pruned_net = 0
	n = 0
	#print 'threshold_prune：',threshold_prune
	#print 'lenv； ', len(layer_original)
	for k,v in solver_pruned.net.params.items():
		zero_exited_layer = 0
		num_pruned_layer = 0
		for i in v:
			#print 'k: ',k,' dim: ',i
			zero_exited_layer += i.data.size-np.count_nonzero(i.data)
			i.data[...] = np.where(abs(i.data) >= threshold_prune[k] , i.data ,0)		
			num_pruned_layer += i.data.size-np.count_nonzero(i.data)-zero_exited_layer
			n +=i.data.size
		#print 'layer_n: ',n
		#print 'zero_exited_layer: ',zero_exited_layer,' rate: ',(zero_exited_layer/float(n))*100.0
		#print 'num_pruned_layer: ',num_pruned_layer,' rate: ',(num_pruned_layer/float(n))*100.0
		zero_exited_net += zero_exited_layer
		num_pruned_net += num_pruned_layer
	#print 'zero_exited_net: ',zero_exited_net,' rate: ',(zero_exited_net/float(n))*100.0
	#print 'num_pruned_net: ',num_pruned_net,' rate: ',(num_pruned_net/float(n))*100.0
	#f.write(str(threshold_prune)+ '\t'+str((num_pruned/float(n))*100.0)+'\n')



## number of testing  iteration is 100*test_it
def test_accuracy(solver_t):
	accuracy_sum = 0
	for test_it in range(100):
		solver_t.test_nets[0].forward()
		accuracy_sum += solver_t.test_nets[0].blobs['accuracy'].data
		#print 'accurracy_blob: ',solver_t.test_nets[0].blobs['accuracy'].data
		#print 'test_accuracy:',solver_t.test_nets[0].blobs['accuracy'].data
	#return solver_t.test_nets[0].blobs['accuracy'].data
		# print 'accuracy: ',solver_t.test_nets[0].blobs['accuracy'].data
		# print 'label argmax: ',solver_t.test_nets[0].blobs['fc8'].data.argmax(1).size
		# print 'sum: ',sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
		# correct += sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
	return accuracy_sum/float(100.0)
# def test_accuracy_ending(solver_t):
# 	accuracy_sum = 0
# 	for test_it in range(50):
# 		solver_t.test_nets[0].forward()
# 		accuracy_sum += solver_t.test_nets[0].blobs['accuracy'].data
# 		# print 'accuracy: ',solver_t.test_nets[0].blobs['accuracy'].data
# 		# print 'label argmax: ',solver_t.test_nets[0].blobs['fc8'].data.argmax(1).size
# 		# print 'sum: ',sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
# 		# correct += sum(solver_t.test_nets[0].blobs['fc8'].data.argmax(1) == solver_t.test_nets[0].blobs['label'].data)
# 	return accuracy_sum/float(50.0)


# def init_testnet_dist(origal_net,solver_net,bound):
# 	dist_params = {}
# 	lenv = len(origal_net.params.items())
# 	n = lenv
# 	for k,v in origal_net.params.items():
# 		z=3
# 		params_dim = 0
# 		layer_mean = np.mean(origal_net.params[k][0].data)
# 		layer_std = np.std(origal_net.params[k][0].data)
# 		dist_params[k]= [layer_mean,layer_std]
# 		bound[k] = [0.0,z*layer_std+layer_mean,n/float(lenv)] # according z=(x-mean)/std
# 		for i in v :
# 			print 'k: ',k,' params_dim: ',params_dim
# 			solver_net.test_nets[0].params[k][params_dim].data[...] = i.data
# 			params_dim += 1
# 		n -= 1
# 	print 'dist_params: ',dist_params

# start = time.time()
# layer_bound = {}
# layer_threshold = {}
# init_testnet_dist(net,solver,layer_bound)
# accuray_based=test_accuracy(solver)
# print layer_bound
# print 'accuray_based: ',accuray_based
# print('initial: %.5f \n' % (time.time()-start))
######weight_prune(net.params['conv1'],solver.test_nets[0].params['conv1'],0.15659839)

######print '15659839: ',test_accuracy(solver)
def statistics(dict_z):
	dict_ori = {}
	zero_exited_net = 0
	n = 0
	#print 'threshold_prune：',threshold_prune
	#print 'lenv； ', len(layer_original)
	for k,v in solver.net.params.items():
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

def train(niter,niter_test):
	print 'accuracy_base: ',test_accuracy(solver)
	for it in range(niter):
		#weight_prune(solver,threshold)
		solver.step(1)
		#params_output_first()
		weight_prune(solver,threshold)
		#params_output()
		#train_loss[it] = solver.net.blobs['loss'].data
		if (it % niter_test ==0):
		# 	print 'accurracy_iter:',it
		# 	test_accuracy(solver)
			print 'iterations:',it
			solver.net.save(output_dir+'/itered_'+str(it)+'.caffemodel')
			print output_dir+'/itered_'+str(it)+'.caffemodel Saved'
			statistics_prune_rate()
			#print 'accurracy_blob: ',solver.test_nets[0].blobs['accuracy'].data
		#	print 'Iteration', it, 'loss = ',solver.net.blobs['loss'].data
		# if it % niter_p ==0:
		# 	print 'Iteration', it, 'testing...'
		# 	print 'pre-prune_acc: ',test_accuracy(solver)
		# 	#weight_prune(solver,threshold)
		# 	#test_acc[it // niter_p] = test_accuracy(solver,threshold)
		# 	print 'pruned_acc: ',test_accuracy(solver)
def params_output_first():
	for k,v in solver.net.params.items():
		params_dim = 0
		for i in v :
			file_dir=os.path.dirname(output_dir+'/' +str(k)+'.txt')
			if(os.path.exists(file_dir)==False):
				os.mkdir(file_dir)
				print 'mkdir:',file_dir
			if(os.path.exists(output_dir+'/'+str(k)+'_param['+str(params_dim)+'].txt')):
				os.remove(output_dir+'/'+str(k)+'_param['+str(params_dim)+'].txt')
			f=open(output_dir+'/'+str(k)+'_param['+str(params_dim)+'].txt','a')
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
def params_output():
	for k,v in solver.net.params.items():
		params_dim = 0
		for i in v :
			file_dir=os.path.dirname(output_dir +'/prune/'+str(k)+'.txt')
			if(os.path.exists(file_dir)==False):
				os.mkdir(file_dir)
				print 'mkdir:',file_dir
			if(os.path.exists(output_dir+'/prune/'+str(k)+'_param['+str(params_dim)+'].txt')):
				os.remove(output_dir+'/prune/'+str(k)+'_param['+str(params_dim)+'].txt')
			f=open(output_dir+'/prune/'+str(k)+'_param['+str(params_dim)+'].txt','a')
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
def statistics_prune_rate():
	statistics(dict_after)
	#print 'dict_net',dict_net
	#print 'dict_pre',dict_pre
	#print 'dict_after',dict_after
	print 'threshold：',threshold
	for k,v in solver.net.params.items():
		print '\n',k,': '
		print 'pre: ',(dict_pre[k]/float(dict_net[k])*100.0)
		print 'aft: ',(dict_after[k]/float(dict_net[k])*100.0)
		print 'up :',(dict_after[k]/float(dict_net[k])*100.0) - (dict_pre[k]/float(dict_net[k])*100.0)
	print '\n','net: '
	print 'pre: ',(dict_pre['net']/float(dict_net['net'])*100.0)
	print 'aft: ',(dict_after['net']/float(dict_net['net'])*100.0)
	print 'up :',(dict_after['net']/float(dict_net['net'])*100.0) - (dict_pre['net']/float(dict_net['net'])*100.0)

def main():
	if(os.path.exists(output_dir)==False):
		os.mkdir(output_dir)
		print 'mkdir:',output_dir
	start = time.time()
	#threshold_pre = threshold = {'conv3': 0.02565, 'conv2': 0.02501, 'conv1': 0.0409, 'ip1': 0.010505613189889115}
	accuracy_prune = test_accuracy(solver)
	weight_prune(solver,threshold)
	accuracy_preiter = test_accuracy(solver)
	solver.net.save(output_dir+'/preiter.caffemodel')
	dict_net_s = statistics(dict_pre)
	for k,v in dict_net_s.items():
		dict_net[k] = v
	#print 'dict_net',dict_net
	#print 'dict_pre',dict_pre
	#params_output_first()
	train(30010,5000)
	statistics_prune_rate()
	print 'accuracy_prune :',accuracy_prune
	print 'accuracy_preiter :',accuracy_preiter
	print 'accuracy_itered: ',test_accuracy(solver)
	# statistics(dict_after)
	# print 'threshold：',threshold
	# for k,v in solver.net.params.items():
	# 	print '\n',k,': '
	# 	print 'pre: ',(dict_pre[k]/float(dict_net[k])*100.0)
	# 	print 'aft: ',(dict_after[k]/float(dict_net[k])*100.0)
	# 	print 'up :',(dict_after[k]/float(dict_net[k])*100.0) - (dict_pre[k]/float(dict_net[k])*100.0)
	# print '\n','net: '
	# print 'pre: ',(dict_pre['net']/float(dict_net['net'])*100.0)
	# print 'aft: ',(dict_after['net']/float(dict_net['net'])*100.0)
	# print 'up :',(dict_after['net']/float(dict_net['net'])*100.0) - (dict_pre['net']/float(dict_net['net'])*100.0)
	# print 'accuracy_prune :',accuracy_prune
	# print 'accuracy_preiter :',accuracy_preiter
	# print 'accuracy_itered: ',test_accuracy(solver)
	#params_output()
	print('Using %.3f s'%(time.time()-start))
	solver.net.save(output_dir+'/itered.caffemodel')


if __name__ == "__main__":
	main()
			


























# for_time = time.time()
# for k,v in reversed(net.params.items()):
# 	# file_dir=os.path.dirname(output_dir +str(k)+'.txt')
# 	# if(os.path.exists(file_dir)==False):
# 	# 	os.mkdir(file_dir)
# 	# 	print 'mkdir:',file_dir
# 	# if(os.path.exists(output_dir+str(k)+'_prunedRate.txt')):
# 	# 	os.remove(output_dir+str(k)+'_prunedRate.txt')
# 	# f=open(output_dir+str(k)+'_prunedRate.txt','a')
# 	print '\n',k,' original: ',solver.test_nets[0].params[k][0].data
# 	print '\n',k,':'
# 	bound_bottom = layer_bound[k][0]
# 	bound_top = layer_bound[k][1]
# 	bound_median = 0
# 	pre_median = np.infty
# 	print ('prune_bottom: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_bottom,k,f))
# 	accuracy_bottom = test_accuracy(solver)
# 	print 'accuracy_bottom: ',accuracy_bottom
# 	weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
# 	print ('prune_top: %d ' % weight_prune(net.params[k],solver.test_nets[0].params[k],bound_top,k,f))
# 	accuracy_top = test_accuracy(solver)
# 	print 'accuracy_top: ',accuracy_top
# 	weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
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
# 		weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
# 		print '\n\nbound_bottom: ',bound_bottom
# 		print 'bound_top: ',bound_top
# 		bound_median = binarySearch(bound_bottom,bound_top)
# 		print 'bound_median: ',bound_median
# 		num = weight_prune(net.params[k],solver.test_nets[0].params[k],bound_median,k,f)
# 		print ('prune_median: %d ' % num )
# 		print 'layer_accuracy_bound: ',0.03*layer_bound[k][2]
# 		print 'n: ',n
# 		accuracy_median = test_accuracy(solver)
# 		print 'accuracy_median: ',accuracy_median
# 		if(accuracy_median < accuray_based - 0.03*layer_bound[k][2]):
# 			bound_top = bound_median
# 			flag = False
# 			print 'bound_top'
# 		elif(accuracy_median >= accuray_based - 0.03*layer_bound[k][2]):
# 			bound_bottom = bound_median
# 			flag = True
# 			print 'bound_bottom'
# 		else:
# 			flag = True
# 		if(n == num and flag):
# 			same = 1
# 		if(pre_median == bound_median):#extrem predition
# 			same =1
# 			bound_median = bound_bottom
# 		pre_median = bound_median
# 		n = num
# 		#weight_recover(net.params[k],solver.test_nets[0].params[k],bound_bottom)
# 		#init_testnet(net,solver)
# 	amount_pruned += num
# 	print k,': prune_num: ',num
# 	print k,': threshold: ',bound_median
# 	layer_threshold[k] = bound_median
# 	amount_weight += net.params[k][0].data.size
# 	amount_weight += net.params[k][1].data.size
# 	print('%s : search_time : %.5f \n' % (k,(time.time()-search_time)))
# 	print '\n',k,' pruned: ',solver.test_nets[0].params[k][0].data
# print 'layer_threshold: ',layer_threshold
# print 'ending accuracy: ',test_accuracy(solver)
# print 'amount_pruned: ',amount_pruned
# print 'amount_weight: ',amount_weight
# print 'pruned_rate:',(amount_pruned/float(amount_weight))*100.0
# print('for_time : %.5f \n' % (time.time()-for_time))
# nsas = os.path.splitext(caffemodel_filename)
# if(os.path.exists(nsas[0]+'_pruned.caffemodel')):
#  	os.remove(nsas[0]+'_pruned.caffemodel')
# solver.test_nets[0].save(nsas[0]+'_pruned.caffemodel')


