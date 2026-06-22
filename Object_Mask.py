import torchvision.transforms as transforms
import copy
import warnings
from Yolov3.mynewdetect import *
from torchvision.datasets.cifar import CIFAR100
import torch
import torch.utils.data as data


def generate_pseudo_anomalies(net_in, yolo_model, cifar_loader, cifar_iter, device='cuda',
							  channel_in=1):
	"""
    接收 cifar_loader 和 cifar_iter。
    在迭代器耗尽时，使用 loader 重置 iter。
    返回新的 iter 状态。
    """
	pseudo_net_in = copy.deepcopy(net_in)
	for sample_idx in range(net_in.shape[0]):
		try:
			cifar_img, _ = next(cifar_iter)
		except StopIteration:
			cifar_iter = iter(cifar_loader)
			cifar_img, _ = next(cifar_iter)

		cifar_img = cifar_img.squeeze(0)
		if channel_in == 1:
			cifar_img_3ch = torch.cat([cifar_img, cifar_img, cifar_img], dim=0)
			cifar_img = cifar_img_3ch

		with torch.no_grad():
			single_normal = copy.deepcopy(net_in[sample_idx])
			yolo_input = LoadTensor(single_normal, channel_in=channel_in)
			single_pseudo, _ = detect(
				model=yolo_model,
				dataset=yolo_input,
				save_img=False,
				cifar_img=cifar_img,
				half=False,
				names='Yolov3/yolov3/data/coco.names',
				output='temp_output',
				conf_thres=0.3,
				iou_thres=0.5,
				device=str(device),
				classes=[0],
				channel_in=channel_in
			)

		pseudo_net_in[sample_idx] = single_pseudo.to(device)

	return pseudo_net_in, cifar_iter


def init_dependencies(cifar_path='dataset/cifar100',
					  yolo_cfg='Yolov3/yolov3/cfg/yolov3-spp.cfg',
					  yolo_weights='Yolov3/yolov3/weights/yolov3-spp-ultralytics.pt',
					  device='cuda',
					  channel_in=1):
	if channel_in == 1:

		cifar_trans_list = [
			transforms.Grayscale(num_output_channels=1),
			transforms.ToTensor()
		]
	else:  # channel_in == 3

		cifar_trans_list = [
			transforms.ToTensor()
		]

	cifar_trans = transforms.Compose(cifar_trans_list)
	cifar_data = CIFAR100(root=cifar_path, train=True, transform=cifar_trans, download=True)

	try:
		cifar_generator = torch.Generator(device='cpu')
	except TypeError:
		cifar_generator = torch.Generator()
	cifar_generator.manual_seed(torch.initial_seed())

	cifar_loader = data.DataLoader(
		cifar_data,
		batch_size=1,
		shuffle=True,
		drop_last=True,
		generator=cifar_generator
	)
	cifar_iter = iter(cifar_loader)

	yolo_model = Darknet(cfg=yolo_cfg)

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=FutureWarning)
		try:
			checkpoint = torch.load(
				yolo_weights,
				map_location=torch.device(device),
				weights_only=False
			)
		except TypeError:
			# PyTorch versions before 2.0 do not expose weights_only.
			checkpoint = torch.load(yolo_weights, map_location=torch.device(device))
		yolo_weights = checkpoint['model']
	yolo_model.load_state_dict(yolo_weights)
	yolo_model.to(device).eval()

	return yolo_model, cifar_loader, cifar_iter
