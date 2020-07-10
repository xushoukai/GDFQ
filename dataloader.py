"""
data loder for loading data
"""
import os
import math
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import struct

__all__ = ["DataLoader", "PartDataLoader"]


class ImageLoader(data.Dataset):
	def __init__(self, dataset_dir, transform=None, target_transform=None):
		class_list = os.listdir(dataset_dir)
		datasets = []
		for cla in class_list:
			cla_path = os.path.join(dataset_dir, cla)
			files = os.listdir(cla_path)
			for file_name in files:
				file_path = os.path.join(cla_path, file_name)
				if os.path.isfile(file_path):
					# datasets.append((file_path, tuple([float(v) for v in int(cla)])))
					datasets.append((file_path, [float(cla)]))
					# print(datasets)
					# assert False
		
		self.dataset_dir = dataset_dir
		self.datasets = datasets
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		frames = []
		
		file_path, label = self.datasets[index]
		noise = torch.load(file_path, map_location=torch.device('cpu'))
		return noise, torch.Tensor(label)
	
	def __len__(self):
		return len(self.datasets)


class DataLoader(object):
	"""
	data loader for CV data sets
	"""
	
	def __init__(self, dataset, batch_size, n_threads=4,
	             ten_crop=False, data_path='/home/dataset/', logger=None):
		"""
		create data loader for specific data set
		:params n_treads: number of threads to load data, default: 4
		:params ten_crop: use ten crop for testing, default: False
		:params data_path: path to data set, default: /home/dataset/
		"""
		self.dataset = dataset
		self.batch_size = batch_size
		self.n_threads = n_threads
		self.ten_crop = ten_crop
		self.data_path = data_path
		self.logger = logger
		self.dataset_root = data_path
		
		self.logger.info("|===>Creating data loader for " + self.dataset)
		
		if self.dataset in ["cifar100"]:
			self.train_loader, self.test_loader = self.cifar(
				dataset=self.dataset)
		
		elif self.dataset in ["imagenet"]:
			self.train_loader, self.test_loader = self.imagenet(
				dataset=self.dataset)
		else:
			assert False, "invalid data set"
	
	def getloader(self):
		"""
		get train_loader and test_loader
		"""
		return self.train_loader, self.test_loader

	def imagenet(self, dataset="imagenet"):

		traindir = os.path.join(self.data_path, "train")
		testdir = os.path.join(self.data_path, "val")

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])

		train_loader = torch.utils.data.DataLoader(
			dsets.ImageFolder(traindir, transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.n_threads,
			pin_memory=True)

		test_transform = transforms.Compose([
			transforms.Resize(256),
			# transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize
		])

		test_loader = torch.utils.data.DataLoader(
			dsets.ImageFolder(testdir, test_transform),
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.n_threads,
			pin_memory=False)
		return train_loader, test_loader

	def cifar(self, dataset="cifar100"):
		"""
		dataset: cifar
		"""
		if dataset == "cifar10":
			norm_mean = [0.49139968, 0.48215827, 0.44653124]
			norm_std = [0.24703233, 0.24348505, 0.26158768]
		elif dataset == "cifar100":
			norm_mean = [0.50705882, 0.48666667, 0.44078431]
			norm_std = [0.26745098, 0.25568627, 0.27607843]
		
		else:
			assert False, "Invalid cifar dataset"

		test_data_root = self.dataset_root

		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(norm_mean, norm_std)])

		if self.dataset == "cifar10":
			test_dataset = dsets.CIFAR10(root=test_data_root,
			                             train=False,
			                             transform=test_transform)
		elif self.dataset == "cifar100":
			test_dataset = dsets.CIFAR100(root=test_data_root,
			                              train=False,
			                              transform=test_transform,
			                              download=True)
		else:
			assert False, "invalid data set"

		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												  batch_size=200,
												  shuffle=False,
												  pin_memory=True,
												  num_workers=self.n_threads)
		return None, test_loader
	

