from __future__ import print_function
import argparse

from koalanet import KOALAnet


def parse_args():
	parser = argparse.ArgumentParser(description="SISR")

	parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
	parser.add_argument('--factor', type=int, default=4, help='scale factor')

	""" Training Settings """
	parser.add_argument('--training_stage', type=int, default=3, choices=[1, 2, 3], help='Set stage for the 3-stage training strategy.')
	parser.add_argument('--tensorboard', type=bool, default=True, help='If set to True, tensorboard summaries are created')
	parser.add_argument('--training_data_path', type=str, default='./dataset/DIV2K/train/DIV2K_train_HR', help='training_dataset path')
	parser.add_argument('--validation_data_path', type=str, default='./dataset/DIV2K/val/DIV2K_valid_HR', help='validation_dataset path')

	""" Testing Settings """
	parser.add_argument('--eval', type=bool, default=True, help='If set to True, evaluation is performed with HR images during the testing phase')
	parser.add_argument('--test_data_path', type=str, default='./testset/Set5/LR/X4/imgs', help='test dataset path')
	parser.add_argument('--test_label_path', type=str, default='./testset/Set5/HR', help='test dataset label path for eval')
	parser.add_argument('--test_ckpt_path', type=str, default='./pretrained_ckpt', help='checkpoint path with trained weights')
	parser.add_argument('--test_patch', type=int, nargs='+', default=[1, 1], help='input image can be divide into an nxn grid of smaller patches in the test phase to fit memory')

	""" Model Settings """ 
	parser.add_argument('--channels', type=int, default=3, help='img channels')
	parser.add_argument('--bicubic_size', type=int, default=20, help='size of bicubic kernel - should be an even number; we recommend at least 4*factor; only 4 centered values are meaningful and other (bicubic_size-4) values are all zeros.')
	parser.add_argument('--gaussian_size', type=int, default=15, help='size of anisotropic gaussian kernel - should be an odd number')
	parser.add_argument('--down_kernel', type=int, default=20, help='downsampling kernel size in the downsampling network')
	parser.add_argument('--up_kernel', type=int, default=5, help='upsampling kernel size in the upsampling network')
	parser.add_argument('--anti_aliasing', type=bool, default=False, help='Matlab anti-aliasing')

	""" Hyperparameters """
	parser.add_argument('--max_epoch', type=int, default=2000, help='number of total epochs')
	parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
	parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation')
	parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
	parser.add_argument('--val_patch_size', type=int, default=100, help='validation patch size')
	parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
	parser.add_argument('--lr_type', type=str, default='stair_decay', choices=['stair_decay', 'linear_decay', 'no_decay'])
	parser.add_argument('--lr_stair_decay_points', type=int, nargs='+', help='stair_decay - Epochs where lr is decayed', default=[1600, 1800])
	parser.add_argument('--lr_stair_decay_factor', type=float, default=0.1, help='stair_decay - lr decreasing factor')
	parser.add_argument('--lr_linear_decay_point', type=int, default=100, help='linear decay - Epoch to start lr decay')
	parser.add_argument('--Qsize', type=int, default=50, help='number of random crop patches from a image')
	parser.add_argument('--n_display', type=int, default=4, help='number images to display - Should be less than or equal to batch_size')
	return parser.parse_args()


def main():
	args = parse_args()
	# set model class
	model = KOALAnet(args)
	# build model
	model.build_model(args)

	# train
	if args.phase == 'train':
		print("Training phase starts!!!")
		model.train(args)
	# test
	elif args.phase == 'test':
		print("Testing phase starts!!!")
		model.test()


if __name__ == '__main__':
	main()
