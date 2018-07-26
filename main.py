from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datetime import datetime
import shutil
import math
from models import _net
import numpy as np
import time
import load_mnist as dset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='what kind of dataset')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--num_classes', type=int, default=10, help='number of')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--max_iter', type=int, default=100000, help='number of iterations to train for') 
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
parser.add_argument('--lr_step', type=int, nargs='+', default=[64000, 96000], help='Learning Rate Decay Steps')  
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--out_digit', type=int, default=9, help='Outlier class')
parser.add_argument('--epsilon', type=float, default=0.0014, help='Magnitude of perturbation for ODIN Detector')
parser.add_argument('--temperature', type=float, default=1000.0, help='Softmax temperature calibration')


parser.add_argument('--test', help='Test when you type --test', action='store_true')
parser.add_argument('--outlier_test', help='Test when you type --outlier_test', action='store_true')
parser.add_argument('--save_dir', type=str, default='logs', help='Directory name to save the checkpoints')
parser.add_argument('--load_dir', type=str, default='', help='Directory name to load checkpoints')
parser.add_argument('--load_pth', type=str, default='', help='pth name to load checkpoints')

opt = parser.parse_args() 
print(opt)

if opt.load_dir: 
	assert os.path.isdir(opt.load_dir)
	opt.save_dir = opt.load_dir
else: 			  		   		
	#opt.save_dir = '{}/{}_{}'.format(opt.save_dir, opt.dataset, datetime.now().strftime("%m%d_%H%M%S"))
	opt.save_dir = '{}/{}_{}'.format(opt.save_dir, opt.dataset, opt.out_digit)
	if opt.test:
		opt.load_dir = opt.save_dir 
		opt.load_pth = 'model_best.pth.tar'
try:
    os.makedirs(opt.save_dir)
except OSError:
    pass
###################################################################################################
# Prepare to train
transform = transforms.Compose(
    # [transforms.RandomCrop(32, padding=4),
	 # transforms.RandomHorizontalFlip(),
	 [transforms.Resize(opt.imageSize),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

num_classes = opt.num_classes

trainset = dset.MNIST(root='./data', train=True, out_digit=opt.out_digit,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=opt.workers)
testset = dset.MNIST(root='./data', train=False, out_digit=opt.out_digit,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers)
outlierset = dset.MNIST(root='./data', train=False, outlier = True, out_digit=opt.out_digit,
                                       download=True, transform=transform)
outlierloader = torch.utils.data.DataLoader(outlierset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers)

nc = 1
net = _net(num_classes, nc)
criterion     = nn.CrossEntropyLoss()

net.cuda()
criterion.cuda()

optimizer = optim.SGD(net.parameters(), 
			lr=opt.lr, momentum=opt.momentum, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step)


if opt.load_dir:
	ckpt = torch.load(opt.load_dir+'/'+opt.load_pth)
	net 	 .load_state_dict(ckpt['state_dict'])
	optimizer.load_state_dict(ckpt['optimizer'])
	scheduler.load_state_dict(ckpt['scheduler'])
	iter_resume = ckpt['iter']
	print('loading network SUCCESSFUL')
else:
	iter_resume = 0
	print('loading network FAILURE')
###################################################################################################
# Start testing
if opt.test:
	net.eval()
	f1 = open("./softmax_scores/inlier.txt", 'w')
	f2 = open("./softmax_scores/outlier.txt", 'w')
	g1 = open("./softmax_scores/inlier_perturb.txt", 'w')
	g2 = open("./softmax_scores/outlier_perturb.txt", 'w')
	#norm_inlier = 0
	test_loss = test_acc = n = 0.0
	for j, test_data in enumerate(testloader, 0):
		imgs, labels = test_data
		imgs = Variable(imgs.cuda(), requires_grad = True); labels = Variable(labels.cuda())
		## Usage
		logits = net(imgs)
		logits = logits / opt.temperature
		nnlogsoftmax = torch.nn.LogSoftmax(dim=1)(logits)

		nnlogsoftmax = nnlogsoftmax.data.cpu().numpy()
		nnsoftmax = np.exp(nnlogsoftmax)
		nnmaxscore = np.max(nnsoftmax, 1)


		for i in range(nnmaxscore.shape[0]):
			f1.write("{}\n".format(nnmaxscore[i]))
		
		predLabel = np.argmax(nnsoftmax, axis=1)
		predLabels = Variable(torch.LongTensor(predLabel).cuda())

		## Save end
		loss = criterion(logits, predLabels)

		## Inverse of adversarial perturbation for single images.
		loss.backward()
		## Gradient for image : imgs.grad.data for compute
		gradient = torch.ge(imgs.grad.data, 0)
		gradient = (gradient.float() - 0.5) * 2    ## -1 to 1 interpolation in gradient

		tempImgs = torch.add(imgs, -opt.epsilon, gradient)
		templogits = net(tempImgs)
		templogits = templogits / opt.temperature

		nntemplogsoftmax = torch.nn.LogSoftmax(dim=1)(templogits)
		## Save
		nntemplogsoftmax = nntemplogsoftmax.data.cpu().numpy()
		nntempsoftmax = np.exp(nntemplogsoftmax)
		nntempmaxscore = np.max(nntempsoftmax, 1)

		#norm = np.sum(abs(gradient))
		#norm_inlier = norm_inlier + norm

		for i in range(nntempmaxscore.shape[0]):
			g1.write("{}\n".format(nntempmaxscore[i]))    
		test_loss += imgs.size(0)*loss.data
		## Usage
		_, pred = torch.max(logits.data, -1)
		acc = float((pred==labels.data).sum())
		test_acc += acc

		n += imgs.size(0)
	print('======inlier confidence file saved')
	test_loss /= n
	test_acc  /= n

	print('####\tTESTING...loss: %5f, acc: %5f' 
		%(test_loss, test_acc))

	if opt.outlier_test:
		#norm_outlier = 0
		n0, n1, n2, n3, n4, n5, n6, n7, n8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
		#net.eval()
		#f1 = open("./softmax_scores/inlier.txt", 'w')
		#f2 = open("./softmax_scores/outlier.txt", 'w')
		#test_loss = test_acc = n = 0.0
		for j, test_data in enumerate(outlierloader, 0):
			imgs, labels = test_data
			imgs = Variable(imgs.cuda(), requires_grad = True); labels = Variable(labels.cuda())
			## Usage
			logits = net(imgs)
			logits = logits / opt.temperature
			nnlogsoftmax = torch.nn.LogSoftmax(dim=1)(logits)

			nnlogsoftmax = nnlogsoftmax.data.cpu().numpy()
			nnsoftmax = np.exp(nnlogsoftmax)
			nnmaxscore = np.max(nnsoftmax, 1)
			nnpred = np.argmax(nnsoftmax, 1)
			
			## Save
			if j == 1:
				img = vutils.make_grid(imgs.data*0.5+0.5)
				vutils.save_image(img, '%s/x.jpg'%(opt.save_dir))
			#nnsoftmax = np.exp(nnlogsoftmax)
			

			n0 = n0 + np.count_nonzero(nnpred==0)
			n1 = n1 + np.count_nonzero(nnpred==1)
			n2 = n2 + np.count_nonzero(nnpred==2)
			n3 = n3 + np.count_nonzero(nnpred==3)
			n4 = n4 + np.count_nonzero(nnpred==4)
			n5 = n5 + np.count_nonzero(nnpred==5)
			n6 = n6 + np.count_nonzero(nnpred==6)
			n7 = n7 + np.count_nonzero(nnpred==7)
			n8 = n8 + np.count_nonzero(nnpred==8)
			
			for i in range(nnmaxscore.shape[0]):
				f2.write("{}\n".format(nnmaxscore[i]))

			predLabel = np.argmax(nnsoftmax, axis=1)
			predLabels = Variable(torch.LongTensor(predLabel).cuda())


			## Save end
			loss = criterion(logits, predLabels)

			loss.backward()

			## Gradient for image : imgs.grad.data for compute
			gradient = torch.ge(imgs.grad.data, 0)
			gradient = (gradient.float() - 0.5) * 2    ## -1 to 1 interpolation in gradient

			tempImgs = torch.add(imgs, -opt.epsilon, gradient)
			templogits = net(tempImgs)
			templogits = templogits / opt.temperature
			if j == 1:
				img2 = vutils.make_grid(tempImgs.data*0.5+0.5)
				vutils.save_image(img2, '%s/x2.jpg'%(opt.save_dir))
				vutils.save_image((img2-img), '%s/x3.jpg'%(opt.save_dir))
			#norm = np.sum(abs(gradient))
			#norm_outlier = norm_outlier + norm
			nntemplogsoftmax = torch.nn.LogSoftmax(dim=1)(templogits)
			## Save
			nntemplogsoftmax = nntemplogsoftmax.data.cpu().numpy()
			nntempsoftmax = np.exp(nntemplogsoftmax)
			nntempmaxscore = np.max(nntempsoftmax, 1)


			for i in range(nntempmaxscore.shape[0]):
				g2.write("{}\n".format(nntempmaxscore[i]))
		print('======outlier confidence file saved')
		print(n0, n1, n2, n3, n4, n5, n6, n7, n8)
		

			#test_loss += imgs.size(0)*loss.data
			## Usage
			#_, pred = torch.max(logits.data, -1)
			#acc = float((pred==labels.data).sum())
			#test_acc += acc

			#n += imgs.size(0)

		#test_loss /= n
		#test_acc  /= n

		#print('\tTESTING...loss: %5f, acc: %5f' 
		#	%(test_loss, test_acc))
	net.train()
###################################################################################################
# Start training
else:
	break_flag = False
	iter = iter_resume
	best_test_acc = 0.0
	for epoch in range(opt.max_iter):
		for i, data in enumerate(trainloader, 0):

			net.zero_grad()
			imgs, labels = data
			imgs = Variable(imgs.cuda()); labels = Variable(labels.cuda()); 

			logits = net(imgs)
			loss = criterion(logits, labels)

			_, pred = torch.max(logits.data, -1)
			acc = float((pred==labels.data).sum()) / imgs.size(0)

			loss.backward()
			optimizer.step()
			scheduler.step() 
	 	    ###########################################################################################
			if iter == 0:
				img = vutils.make_grid(imgs.data*0.5+0.5)
				vutils.save_image(img, '%s/x.jpg'%(opt.save_dir))
				print('%s/x.jpg saved'%(opt.save_dir))

			if iter % 20 == 0:
				print('[%6d/%6d] loss: %5f, acc: %5f, lr: %5f' %(iter, opt.max_iter, loss.data[0], acc, scheduler.get_lr()[0]))

			if iter % 1000 == 0:
				net.eval()
				test_loss = test_acc = n = 0.0
				for j, test_data in enumerate(testloader, 0):
					imgs, labels = test_data
					imgs = Variable(imgs.cuda()); labels = Variable(labels.cuda())

					logits = net(imgs)
					loss = criterion(logits, labels)
					test_loss += imgs.size(0)*loss.data

					_, pred = torch.max(logits.data, -1)
					acc = float((pred==labels.data).sum())
					test_acc += acc

					n += imgs.size(0)

				test_loss /= n
				test_acc  /= n
			#######################################################################################
				print('\tTESTING...loss: %5f, acc: %5f, best_acc: %5f' 
					%(test_loss, test_acc, best_test_acc))
				net.train()

				is_best = test_acc > best_test_acc
				best_test_acc = max(test_acc, best_test_acc)
				state = ({
					'iter': iter,
					'state_dict': net.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict()
					})
				print('saving model...')
				fn = os.path.join(opt.save_dir, 'checkpoint.pth.tar')
				torch.save(state, fn)
				if is_best: 
					fn_best = os.path.join(opt.save_dir, 'model_best.pth.tar')
					print('saving best model...')
					shutil.copyfile(fn, fn_best)

			#if iter == opt.max_iter:
			#	break_flag = True
			#	break
			iter  	 += 1
		###########################################################################################
		if iter >= opt.max_iter:
			break
		if break_flag:
			break

if opt.outlier_test:
	## Baseline
	print('======tpr 95 threshold calculating...')
	inlier = []
	outlier = []
	f1 = open("./softmax_scores/inlier.txt", 'r')
	f2 = open("./softmax_scores/outlier.txt", 'r')
	while True:
		temp = f1.readline()
		if not temp:
			break
		inlier.append(temp[:-2])
	while True:
		temp = f2.readline()
		if not temp:
			break
		outlier.append(temp[:-2])
	f1.close()
	f2.close()


	inlier = np.asarray(inlier, dtype=np.float32)[:, None]
	outlier = np.asarray(outlier, dtype=np.float32)[:, None]
	fpr = 0.0
	count = 0

	inlier_sort = np.sort(inlier, axis=None)
	start_th = (inlier_sort[int(math.floor(len(inlier)*0.045))-1])
	end_th = (inlier_sort[int(math.floor(len(inlier)*0.055))+1])


	for delta in np.arange(start_th, end_th, 0.9/100000):
		tpr = np.sum(np.sum(inlier >= delta)) / np.float(len(inlier))
		#print(tpr)
		if tpr <= 0.955 and tpr >= 0.945:
			fpr_temp = np.sum(np.sum(outlier >= delta)) / np.float(len(outlier))
			fpr = fpr + fpr_temp
			count = count + 1
			delta_real = delta
	fpr = fpr / count
	print('softmax threshold is %5f' %(delta_real))
	print('tpr 95: fpr is %5f' %(fpr))
	


	## ODIN detector for outlier detection
	print('======tpr 95 threshold calculating (ODIN)...')
	inlier_perturb = []
	outlier_perturb = []
	g1 = open("./softmax_scores/inlier_perturb.txt", 'r')
	g2 = open("./softmax_scores/outlier_perturb.txt", 'r')
	while True:
		temp = g1.readline()
		if not temp:
			break
		inlier_perturb.append(temp[:-2])
	while True:
		temp = g2.readline()
		if not temp:
			break
		outlier_perturb.append(temp[:-2])
	g1.close()
	g2.close()


	inlier_perturb = np.asarray(inlier_perturb, dtype=np.float32)[:, None]
	outlier_perturb = np.asarray(outlier_perturb, dtype=np.float32)[:, None]
	fpr = 0.0
	count = 0
	for delta in np.arange(1, 0.1, -0.9/100000):
		tpr = np.sum(np.sum(inlier_perturb >= delta)) / np.float(len(inlier_perturb))
		#print(tpr)
		if tpr <= 0.955 and tpr >= 0.945:
			fpr_temp = np.sum(np.sum(outlier_perturb >= delta)) / np.float(len(outlier_perturb))
			fpr = fpr + fpr_temp
			count = count + 1
			delta_real = delta
	fpr = fpr / count
	print('softmax threshold of ODIN is %5f' %(delta_real))
	print('tpr 95: fpr of ODIN is %5f' %(fpr))
	save_dir = 'softmax_scores/epsilon{}.temperature{}.fpr{}/'.format(opt.epsilon, opt.temperature, fpr)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)



#	print('=====softmax score change thresholding')
#	fpr = 0.0
#	count = 0
	inlier = inlier_perturb - inlier
	outlier = outlier_perturb - outlier
	print(np.mean(inlier))
	print(np.mean(outlier))
#	print(inlier)
#	#time.sleep(10000)
#	inlier_sort = np.sort(inlier, axis=None)
#	print(inlier_sort)
#	#time.sleep(10000)
#	start_th = (inlier_sort[int(math.floor(len(inlier_sort)*0.045))-1])
#	end_th = (inlier_sort[int(math.floor(len(inlier_sort)*0.055))+1])
#	print(start_th, end_th)
#	print(np.count_nonzero(inlier_sort))
#
#	for delta in np.arange(start_th, end_th+0.9/1000000, 0.9/1000000):
#		tpr = np.sum(np.sum(inlier >= delta)) / np.float(len(inlier))
#		#print(tpr)
#		if tpr <= 0.955 and tpr >= 0.945:
#			fpr_temp = np.sum(np.sum(outlier >= delta)) / np.float(len(outlier))
#			fpr = fpr + fpr_temp
#			count = count + 1
#			delta_real = delta
	#fpr = fpr / count
	#print('Delta softmax threshold of ODIN is %5f' %(delta_real))
	#print('tpr 95: fpr of ODIN delta is %5f' %(fpr))
	#try:
	#	os.makedirs('softmax_scores/epsilon{}.temperature{}.fpr{}/'.format(opt.epsilon, opt.temperature, fpr))