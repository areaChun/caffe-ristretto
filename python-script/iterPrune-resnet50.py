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
#resnet50
#threshold = {'res2b_branch2a': 0.10243088891729712, 'res2b_branch2c': 0.01429, 'res2b_branch2b': 0.096193867939291522, 'bn4c_branch2a': 0.31561, 'scale2c_branch2a': 0.0138, 'bn4c_branch2c': 0.00317, 'bn4c_branch2b': 0.05562, 'bn2a_branch2a': 0.04889, 'bn2a_branch2c': 0.0012, 'bn2a_branch2b': 0.05569, 'res2a_branch1': 0.02119, 'scale5c_branch2c': 2.5962553322315216, 'scale5c_branch2b': 1.3645316511392593, 'scale5c_branch2a': 1.2845093011856079, 'scale2c_branch2b': 0.05533, 'bn5b_branch2c': 0.00321, 'bn5b_branch2b': 0.06603, 'bn5b_branch2a': 0.34539097547531128, 'res4a_branch2b': 0.00579, 'res4a_branch2c': 0.063865382457152009, 'scale2b_branch2a': 0.76768, 'res4a_branch2a': 0.061868511897046119, 'bn3b_branch2a': 0.40396, 'bn3b_branch2c': 0.00294, 'bn3b_branch2b': 0.17657, 'scale5a_branch1': 4.0437153577804565, 'scale2c_branch2c': 2.2905756235122681, 'bn3d_branch2c': 0.00643, 'res3b_branch2a': 0.069505520747043192, 'res3b_branch2b': 0.062858190678525716, 'res3b_branch2c': 0.074767493177205324, 'bn3d_branch2a': 0.07952, 'bn4b_branch2b': 0.10041, 'bn4b_branch2c': 0.00503, 'bn4b_branch2a': 0.36688, 'res5b_branch2b': 0.03233656520023942, 'res5b_branch2c': 0.042309126176405698, 'res5b_branch2a': 0.042104548774659634, 'res4a_branch1': 0.004, 'scale4e_branch2a': 1.3940325528383255, 'scale4e_branch2b': 1.6018023788928986, 'scale4e_branch2c': 1.5128187239170074, 'scale3a_branch2c': 2.5232237577438354, 'bn4d_branch2a': 0.22081, 'bn4d_branch2b': 0.07131, 'bn4d_branch2c': 0.00232, 'bn5a_branch1': 0.06294, 'scale4b_branch2a': 1.4622013419866562, 'scale4b_branch2c': 1.4398237764835358, 'scale4b_branch2b': 1.611856073141098, 'res2a_branch2a': 0.20807111449539661, 'bn3c_branch2b': 0.02437, 'bn3c_branch2c': 0.00208, 'scale3d_branch2a': 2.042535126209259, 'bn5a_branch2b': 0.11622, 'bn5a_branch2c': 0.00353, 'bn5a_branch2a': 0.47695, 'res4d_branch2a': 0.050702175009064376, 'scale3d_branch2b': 0.42819, 'res4d_branch2c': 0.049971894826740026, 'res4d_branch2b': 0.042273160477634519, 'res3a_branch1': 0.0102, 'scale4f_branch2a': 1.4767800867557526, 'bn3d_branch2b': 0.1514, 'scale4f_branch2c': 1.4568544030189514, 'scale4f_branch2b': 1.8355064988136292, 'bn4a_branch1': 0.00898, 'bn4e_branch2c': 0.00266, 'bn4e_branch2b': 0.05911, 'bn4e_branch2a': 0.40279, 'scale3d_branch2c': 1.9500834047794342, 'res2a_branch2b': 0.091830148594453931, 'bn5c_branch2a': 0.33872795104980469, 'bn5c_branch2b': -0.07895921915769577, 'bn5c_branch2c': 0.049908600747585297, 'scale3a_branch2b': 1.718206599354744, 'scale4d_branch2c': 1.6543397307395935, 'scale4d_branch2b': 1.5210876017808914, 'scale4d_branch2a': 1.3150766640901566, 'res2a_branch2c': 0.11900905525544658, 'scale3a_branch2a': 0.46148, 'bn4a_branch2c': 0.00641, 'bn4a_branch2b': 0.25471, 'bn4a_branch2a': 0.26083, 'res4c_branch2a': 0.052330321865156293, 'bn2b_branch2a': 0.21931, 'res4c_branch2b': 0.042177025752607733, 'res4c_branch2c': 0.052854371024295688, 'bn_conv1': 0.39319, 'res4e_branch2b': 0.041945040342397988, 'res4e_branch2c': 0.05156942130997777, 'res4e_branch2a': 0.05887473642360419, 'conv1': 0.0015, 'scale4a_branch2a': 0.40488, 'res2c_branch2b': 0.01798, 'res2c_branch2c': 0.10492618661373854, 'res2c_branch2a': 0.02689, 'res5c_branch2a': 0.051730164617765695, 'res4f_branch2a': 0.063303116330644116, 'res5c_branch2c': 0.043721256544813514, 'res5c_branch2b': 0.030612157133873552, 'res3a_branch2a': 0.089342784252949059, 'bn3a_branch2a': 0.06518, 'res3a_branch2c': 0.08682938024867326, 'bn3a_branch2c': 0.00413, 'bn2b_branch2c': 0.00011, 'res5a_branch2b': 0.035190761613193899, 'scale4c_branch2b': 1.5876603424549103, 'scale4c_branch2c': 1.3299859017133713, 'scale4c_branch2a': 1.4361559748649597, 'scale2a_branch2b': 0.19297, 'scale2a_branch2c': 3.0382670760154724, 'scale2a_branch2a': 2.3044238686561584, 'scale3c_branch2a': 1.3694176524877548, 'scale3c_branch2c': 2.3510231971740723, 'scale3c_branch2b': 1.7437249720096588, 'bn2c_branch2c': 0.00623, 'bn2c_branch2b': 0.1309, 'bn2c_branch2a': 0.00727, 'res3c_branch2c': 0.061479904688894749, 'res3c_branch2b': 0.056393142731394619, 'res3c_branch2a': 0.058389602447277866, 'res4b_branch2c': 0.05360613513039425, 'res4b_branch2b': 0.040638741920702159, 'res4b_branch2a': 0.04408784385304898, 'fc1000': 0.10061942534125023, 'scale2b_branch2c': 2.2290872931480408, 'scale2b_branch2b': 0.01237, 'res4f_branch2c': 0.053305963636375964, 'res4f_branch2b': 0.042925508751068264, 'scale4a_branch2b': 1.8594232499599457, 'scale4a_branch2c': 2.1573084890842438, 'bn3a_branch2b': 0.16294, 'res3a_branch2b': 0.01283, 'scale3b_branch2b': 1.866358757019043, 'scale3b_branch2c': 1.9250560104846954, 'scale3b_branch2a': 1.3884455859661102, 'scale3a_branch1': 0.07353, 'scale2a_branch1': 0.10493, 'scale_conv1': 0.41156, 'scale5b_branch2a': 1.2742084711790085, 'scale5b_branch2b': 1.2822731286287308, 'scale5b_branch2c': 2.1083731651306152, 'bn3c_branch2a': 0.16671, 'bn3a_branch1': 0.01362, 'bn2a_branch1': 0.00179, 'scale4a_branch1': 0.07103, 'bn4f_branch2b': 0.03379, 'bn4f_branch2c': 0.00885, 'bn2b_branch2b': 0.16335, 'bn4f_branch2a': 0.08785, 'res5a_branch2c': 0.048276899906340986, 'res5a_branch1': 0.04044971115217777, 'res5a_branch2a': 0.052932568243704736, 'res3d_branch2b': 0.068810177035629749, 'res3d_branch2c': 0.0778878228738904, 'res3d_branch2a': 0.096660210518166423, 'scale5a_branch2a': 1.3766958713531494, 'scale5a_branch2c': 2.5783365070819855, 'scale5a_branch2b': 1.536960706114769}

threshold = {
'conv1': 0.0079578733407 ,\
'bn_conv1': 0.0 ,\
'scale_conv1': 0.0 ,\
'res2a_branch1': 0.0074222042337 ,\
'bn2a_branch1': 0 ,\
'scale2a_branch1': 0.0 ,\
'res2a_branch2a': 0.0083096314818 ,\
'bn2a_branch2a': 0 ,\
'scale2a_branch2a': 0.0 ,\
'res2a_branch2b': 0.0059319902975 ,\
'bn2a_branch2b': 0 ,\
'scale2a_branch2b': 0.0 ,\
'res2a_branch2c': 0.0071608681665 ,\
'bn2a_branch2c': 0 ,\
'scale2a_branch2c': 0.0 ,\
'res2b_branch2a': 0.0087963765524 ,\
'bn2b_branch2a': 0 ,\
'scale2b_branch2a': 0.0 ,\
'res2b_branch2b': 0.0124050863262 ,\
'bn2b_branch2b': 0 ,\
'scale2b_branch2b': 0.0 ,\
'res2b_branch2c': 0.0079182728715 ,\
'bn2b_branch2c': 0 ,\
'scale2b_branch2c': 0.0 ,\
'res2c_branch2a': 0.0129904269706 ,\
'bn2c_branch2a': 0 ,\
'scale2c_branch2a': 0.0 ,\
'res2c_branch2b': 0.0126944702024 ,\
'bn2c_branch2b': 0 ,\
'scale2c_branch2b': 0.0 ,\
'res2c_branch2c': 0.0073305008039 ,\
'bn2c_branch2c': 0 ,\
'scale2c_branch2c': 0.0 ,\
'res3a_branch1': 0.00771076807473 ,\
'bn3a_branch1': 0 ,\
'scale3a_branch1': 0.0 ,\
'res3a_branch2a': 0.0108820007415 ,\
'bn3a_branch2a': 0 ,\
'scale3a_branch2a': 0.0 ,\
'res3a_branch2b': 0.0084991108859 ,\
'bn3a_branch2b': 0 ,\
'scale3a_branch2b': 0.0 ,\
'res3a_branch2c': 0.0076518632965 ,\
'bn3a_branch2c': 0 ,\
'scale3a_branch2c': 0.0 ,\
'res3b_branch2a': 0.00842869600747 ,\
'bn3b_branch2a': 0 ,\
'scale3b_branch2a': 0.0 ,\
'res3b_branch2b': 0.00767944819527 ,\
'bn3b_branch2b': 0 ,\
'scale3b_branch2b': 0.0 ,\
'res3b_branch2c': 0.00704685268179 ,\
'bn3b_branch2c': 0 ,\
'scale3b_branch2c': 0.0 ,\
'res3c_branch2a': 0.00476783324254 ,\
'bn3c_branch2a': 0 ,\
'scale3c_branch2a': 0.0 ,\
'res3c_branch2b': 0.00414663447579 ,\
'bn3c_branch2b': 0 ,\
'scale3c_branch2b': 0.0 ,\
'res3c_branch2c': 0.0047774059251 ,\
'bn3c_branch2c': 0 ,\
'scale3c_branch2c': 0.0 ,\
'res3d_branch2a': 0.0119258861523 ,\
'bn3d_branch2a': 0 ,\
'scale3d_branch2a': 0.0 ,\
'res3d_branch2b': 0.00886458549649 ,\
'bn3d_branch2b': 0 ,\
'scale3d_branch2b': 0.0 ,\
'res3d_branch2c': 0.00634339743108 ,\
'bn3d_branch2c': 0 ,\
'scale3d_branch2c': 0.0 ,\
'res4a_branch1': 0.00732770680333 ,\
'bn4a_branch1': 0 ,\
'scale4a_branch1': 0.0 ,\
'res4a_branch2a': 0.00771793165477 ,\
'bn4a_branch2a': 0 ,\
'scale4a_branch2a': 0.0 ,\
'res4a_branch2b': 0.00560121624731 ,\
'bn4a_branch2b': 0 ,\
'scale4a_branch2b': 0.0 ,\
'res4a_branch2c': 0.00808232636191 ,\
'bn4a_branch2c': 0 ,\
'scale4a_branch2c': 0.0 ,\
'res4b_branch2a': 0.00522708229255 ,\
'bn4b_branch2a': 0 ,\
'scale4b_branch2a': 0.0 ,\
'res4b_branch2b': 0.00493633064907 ,\
'bn4b_branch2b': 0 ,\
'scale4b_branch2b': 0.0 ,\
'res4b_branch2c': 0.00634025234031 ,\
'bn4b_branch2c': 0 ,\
'scale4b_branch2c': 0.0 ,\
'res4c_branch2a': 0.00644219410606 ,\
'bn4c_branch2a': 0 ,\
'scale4c_branch2a': 0.0 ,\
'res4c_branch2b': 0.00486348603154 ,\
'bn4c_branch2b': 0 ,\
'scale4c_branch2b': 0.0 ,\
'res4c_branch2c': 0.00608898024075 ,\
'bn4c_branch2c': 0 ,\
'scale4c_branch2c': 0.0 ,\
'res4d_branch2a': 0.00598341810983 ,\
'bn4d_branch2a': 0 ,\
'scale4d_branch2a': 0.0 ,\
'res4d_branch2b': 0.00483746159589 ,\
'bn4d_branch2b': 0 ,\
'scale4d_branch2b': 0.0 ,\
'res4d_branch2c': 0.00523882722482 ,\
'bn4d_branch2c': 0 ,\
'scale4d_branch2c': 0.0 ,\
'res4e_branch2a': 0.00720144293737 ,\
'bn4e_branch2a': 0 ,\
'scale4e_branch2a': 0.0 ,\
'res4e_branch2b': 0.00498938683886 ,\
'bn4e_branch2b': 0 ,\
'scale4e_branch2b': 0.0 ,\
'res4e_branch2c': 0.00531900757924 ,\
'bn4e_branch2c': 0 ,\
'scale4e_branch2c': 0.0 ,\
'res4f_branch2a': 0.00813226289465 ,\
'bn4f_branch2a': 0 ,\
'scale4f_branch2a': 0 ,\
'res4f_branch2b': 0.00532999491552 ,\
'bn4f_branch2b': 0 ,\
'scale4f_branch2b': 0 ,\
'res4f_branch2c': 0.005838970677 ,\
'bn4f_branch2c': 0 ,\
'scale4f_branch2c': 0 ,\
'res5a_branch1': 0.00529456923978 ,\
'bn5a_branch1': 0 ,\
'scale5a_branch1': 0 ,\
'res5a_branch2a': 0.00611638051923 ,\
'bn5a_branch2a': 0 ,\
'scale5a_branch2a': 0 ,\
'res5a_branch2b': 0.00393669545883 ,\
'bn5a_branch2b': 0 ,\
'scale5a_branch2b': 0 ,\
'res5a_branch2c': 0.00594708599383 ,\
'bn5a_branch2c': 0 ,\
'scale5a_branch2c': 0 ,\
'res5b_branch2a': 0.0045379396528 ,\
'bn5b_branch2a': 0 ,\
'scale5b_branch2a': 0 ,\
'res5b_branch2b': 0.00331467082724 ,\
'bn5b_branch2b': 0 ,\
'scale5b_branch2b': 0 ,\
'res5b_branch2c': 0.0052585668047 ,\
'bn5b_branch2c': 0 ,\
'scale5b_branch2c': 0 ,\
'res5c_branch2a': 0.00608548346208 ,\
'bn5c_branch2a': 0 ,\
'scale5c_branch2a': 0 ,\
'res5c_branch2b': 0.00324515871471 ,\
'bn5c_branch2b': 0 ,\
'scale5c_branch2b': 0 ,\
'res5c_branch2c': 0.00543299573474 ,\
'bn5c_branch2c': 0 ,\
'scale5c_branch2c': 0 ,\
'fc1000': 0.0374162505118
}

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
	file_dir=os.path.dirname(output_dir +'/prune_rate.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir +'/prune_rate.txt')):
		os.remove(output_dir+'/prune_rate.txt')
	f=open(output_dir +'/prune_rate.txt','a')
	for k,v in solver.net.params.items():
		print '\n',k,': '
		print 'threshold: ',threshold[k]
		print 'pre: ',(dict_pre[k]/float(dict_net[k])*100.0)
		print 'aft: ',(dict_after[k]/float(dict_net[k])*100.0)
		f.write(str((dict_pre[k]/float(dict_net[k])*100.0))+'\n')
		print 'up :',(dict_after[k]/float(dict_net[k])*100.0) - (dict_pre[k]/float(dict_net[k])*100.0)
	print '\n','net: '
	print 'pre: ',(dict_pre['net']/float(dict_net['net'])*100.0)
	print 'aft: ',(dict_after['net']/float(dict_net['net'])*100.0)
	f.write(str((dict_pre['net']/float(dict_net['net'])*100.0))+'\n')
	print 'up :',(dict_after['net']/float(dict_net['net'])*100.0) - (dict_pre['net']/float(dict_net['net'])*100.0)
	file_dir=os.path.dirname(output_dir +'/layers.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir +'/layers.txt')):
		os.remove(output_dir+'/layers.txt')
	f=open(output_dir +'/layers.txt','a')
	for k,v in solver.net.params.items():
		f.write(str(k)+'\n')
	file_dir=os.path.dirname(output_dir +'/threshold.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir +'/threshold.txt')):
		os.remove(output_dir+'/threshold.txt')
	f=open(output_dir +'/threshold.txt','a')
	for k,v in solver.net.params.items():
		f.write(str(threshold[k])+'\n')
	file_dir=os.path.dirname(output_dir +'/setting.txt')
	if(os.path.exists(file_dir)==False):
		os.mkdir(file_dir)
		print 'mkdir:',file_dir
	if(os.path.exists(output_dir +'/setting.txt')):
		os.remove(output_dir+'/setting.txt')
	f=open(output_dir +'/setting.txt','a')
	for k,v in solver.net.params.items():
		f.write('\''+str(k)+'\': '+str(threshold[k])+' ,\\'+'\n')
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
	train(40000,5000)
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


