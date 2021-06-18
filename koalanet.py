from __future__ import print_function
from datetime import datetime
import time
from tensorflow.data.experimental import shuffle_and_repeat, unbatch

from utils import *
from ops import *


class KOALAnet:
	def __init__(self, args):
		self.phase = args.phase
		self.factor = args.factor

		""" Training Settings """
		self.training_stage = args.training_stage
		self.tensorboard = args.tensorboard

		""" Testing Settings """
		self.eval = args.eval
		self.test_data_path = args.test_data_path
		self.test_label_path = args.test_label_path
		self.test_ckpt_path = args.test_ckpt_path
		self.test_patch = args.test_patch

		""" Model Settings """ 
		self.channels = args.channels
		self.bicubic_size = args.bicubic_size
		self.gaussian_size = args.gaussian_size
		self.down_kernel = args.down_kernel
		self.up_kernel = args.up_kernel
		self.anti_aliasing = args.anti_aliasing

		""" Hyperparameters """
		self.max_epoch = args.max_epoch
		self.batch_size = args.batch_size
		self.val_batch_size = args.val_batch_size
		self.patch_size = args.patch_size
		self.val_patch_size = args.val_patch_size
		self.lr = args.lr
		self.lr_type = args.lr_type
		self.lr_stair_decay_points = args.lr_stair_decay_points
		self.lr_stair_decay_factor = args.lr_stair_decay_factor
		self.lr_linear_decay_point = args.lr_linear_decay_point
		self.n_display = args.n_display

		if self.training_stage == 1:
			self.model_name = 'downsampling_network'
		elif self.training_stage == 2:
			self.model_name = 'upsampling_network_baseline'
		elif self.training_stage == 3:
			self.model_name = 'upsampling_network'

		""" Directories """
		self.ckpt_dir = os.path.join('ckpt', self.model_dir)
		self.result_dir = os.path.join('results')
		check_folder(self.ckpt_dir)
		check_folder(self.result_dir)

		""" Model Init """
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

		""" Print Model """
		print('Model arguments, [{:s}]'.format((str(datetime.now())[:-7])))
		for arg in vars(args):
			print('# {} : {}'.format(arg, getattr(args, arg)))
		print("\n")

	def upsampling_network_baseline(self, input_LR, factor, kernel, channels=3, reuse=False, scope='SISR_DUF'):
		with tf.variable_scope(scope, reuse=reuse):
			ch = 64
			n_res = 12
			net = conv2d(input_LR, ch, 3)
			for res in range(n_res):
				net = res_block(net, ch, 3, scope='Residual_block0_' + str(res + 1))
			net = tf.nn.relu(net)
			# upsampling kernel branch
			k2d = tf.nn.relu(conv2d(net, ch * 2, 3))
			k2d = conv2d(k2d, kernel * kernel * factor * factor, 3)
			# rgb residual image branch
			rgb = tf.nn.relu(conv2d(net, ch * 2, 3))
			rgb = tf.depth_to_space(rgb, 2)
			if factor == 4:
				rgb = tf.nn.relu(conv2d(rgb, ch, 3))
				rgb = tf.depth_to_space(rgb, 2)
			rgb = conv2d(rgb, channels, 3)
			# local filtering and upsampling
			output_k2d = local_conv_us(input_LR, k2d, factor, channels, kernel)
			output = output_k2d + rgb
		return output

	def upsampling_network(self, input_LR, k2d_ds, factor, kernel, channels=3, reuse=False, scope='SISR_DUF'):
		with tf.variable_scope(scope, reuse=reuse):
			ch = 64
			n_res = 12
			skip_idx = np.arange(0, 5, 1)
			# extract degradation kernel features
			k = cr_block(k2d_ds, 3, ch, 3, 'kernel_condition')
			net = conv2d(input_LR, ch, 3)
			filter_p_list = []
			for res in range(n_res):
				if res in skip_idx:
					net, filter_p = koala(net, k, ch, ch, conv_k_sz=3, lc_k_sz=7, scope_res='Residual_block0_' + str(res + 1), scope='KOALA_module/%d' % (res+1))
					filter_p_list.append(filter_p)
				else:
					net = res_block(net, ch, 3, scope='Residual_block0_' + str(res + 1))
			net = tf.nn.relu(net)
			# upsampling kernel branch
			k2d = tf.nn.relu(conv2d(net, ch * 2, 3))
			k2d = conv2d(k2d, kernel * kernel * factor * factor, 3)
			# rgb residual image branch
			rgb = tf.nn.relu(conv2d(net, ch * 2, 3))
			rgb = tf.depth_to_space(rgb, 2)
			if factor == 4:
				rgb = tf.nn.relu(conv2d(rgb, ch, 3))
				rgb = tf.depth_to_space(rgb, 2)
			rgb = conv2d(rgb, channels, 3)
			# local filtering and upsampling
			output_k2d = local_conv_us(input_LR, k2d, factor, channels, kernel)
			output = output_k2d + rgb
		return output, filter_p_list[-1]

	def downsampling_network(self, input_LR, kernel, reuse=False, scope='SISR'):
		with tf.variable_scope(scope, reuse=reuse):
			ch = 64
			skip = dict()
			# encoder
			n, skip[0] = enc_level_res(input_LR, ch, scope='enc_block_res/0')
			n, skip[1] = enc_level_res(n, ch*2, scope='enc_block_res/1')
			# bottleneck
			n = bottleneck_res(n, ch*4)
			# decoder
			n = dec_level_res(n, skip[1], ch*2, scope='dec_block_res/0')
			n = dec_level_res(n, skip[0], ch, scope='dec_block_res/1')
			# downsampling kernel branch
			n = tf.nn.relu(conv2d(n, ch, 3))
			k2d = conv2d(n, kernel * kernel, 3)
		return k2d

	def build_model(self, args):
		data = SISRData(args)
		if self.phase == 'train':
			""" Directories """
			self.log_dir = os.path.join('logs', self.model_dir)
			self.img_dir = os.path.join(self.result_dir, 'imgs_train', self.model_dir)
			check_folder(self.log_dir)
			check_folder(self.img_dir)

			self.updates_per_epoch = int(data.num_train / self.batch_size)
			print("Update per epoch : ", self.updates_per_epoch)

			""" Training Data Generation """
			train_folder_path = tf.data.Dataset.from_tensor_slices(data.list_train).apply(shuffle_and_repeat(len(data.list_train)))
			train_data = train_folder_path.map(data.image_processing, num_parallel_calls=4)
			train_data = train_data.apply(unbatch()).shuffle(data.Qsize*50).batch(data.batch_size).prefetch(1)
			train_data_iterator = train_data.make_one_shot_iterator()

			# self.train_hr : [B, H, W, C], self.gaussian_kernel : [B, gaussian_size, gaussian_size, 1], data.bicubic_kernel : [1, bicubic_size, bicubic_size, B]
			self.train_hr, self.gaussian_kernel = train_data_iterator.get_next()
			self.ds_kernel = get_ds_kernel(data.bicubic_kernel, self.gaussian_kernel)
			self.train_lr = get_ds_input(self.train_hr, self.ds_kernel, self.channels, self.batch_size, data.pad_left, data.pad_right, self.factor)
			self.train_lr = tf.math.round((self.train_lr+1.0)/2.0*255.0)
			self.train_lr = tf.cast(self.train_lr, tf.float32)/255.0 * 2.0 - 1.0
			print("#### Degraded train_lr is quantized.")

			# set placeholders for validation
			self.val_hr = tf.placeholder(tf.float32, (self.val_batch_size, self.val_patch_size * self.factor, self.val_patch_size * self.factor, self.channels))
			self.val_base_k = tf.placeholder(tf.float32, (1, self.bicubic_size, self.bicubic_size, self.val_batch_size))
			self.val_rand_k = tf.placeholder(tf.float32, (self.val_batch_size, self.gaussian_size, self.gaussian_size, 1))
			self.ds_kernel_val = get_ds_kernel(self.val_base_k, self.val_rand_k)
			self.val_lr = get_ds_input(self.val_hr, self.ds_kernel_val, self.channels, self.val_batch_size, data.pad_left, data.pad_right, self.factor)
			self.val_lr = tf.math.round((self.val_lr+1.0)/2.0*255.0)
			self.val_lr = tf.cast(self.val_lr, tf.float32)/255.0 * 2.0 - 1.0
			print("#### Degraded val_lr is quantized.")
			self.list_val = data.list_val
			print("Training patch size : ", self.train_lr.get_shape())
			
			""" Define Model """
			if self.training_stage == 1:
				self.k2d_ds = self.downsampling_network(self.train_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
				self.k2d_ds_val = self.downsampling_network(self.val_lr, self.down_kernel, reuse=True, scope='SISR_DDF')
				# reconstructed LR images
				self.output_ds_hr = local_conv_ds(self.train_hr, self.k2d_ds, self.factor, self.channels, self.down_kernel)
				self.output_ds_hr_val = local_conv_ds(self.val_hr, self.k2d_ds_val, self.factor, self.channels, self.down_kernel)
			elif self.training_stage == 2:
				# reconstructed HR images
				self.output = self.upsampling_network_baseline(self.train_lr, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')
				self.output_val = self.upsampling_network_baseline(self.val_lr, self.factor, self.up_kernel, self.channels, reuse=True, scope='SISR_DUF')
			elif self.training_stage == 3:
				self.k2d_ds = self.downsampling_network(self.train_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
				self.k2d_ds_val = self.downsampling_network(self.val_lr, self.down_kernel, reuse=True, scope='SISR_DDF')
				# reconstructed LR images
				self.output_ds_hr = local_conv_ds(self.train_hr, self.k2d_ds, self.factor, self.channels, self.down_kernel)
				self.output_ds_hr_val = local_conv_ds(self.val_hr, self.k2d_ds_val, self.factor, self.channels, self.down_kernel)
				# reconstructed HR images
				self.output, self.filter_p = self.upsampling_network(self.train_lr, self.k2d_ds, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')
				self.output_val, _ = self.upsampling_network(self.val_lr, self.k2d_ds_val, self.factor, self.up_kernel, self.channels, reuse=True, scope='SISR_DUF')

			""" Define Losses """
			if self.training_stage == 1:
				# training
				self.rec_loss_ds_hr = l1_loss(self.train_lr, self.output_ds_hr)
				self.k2d_ds = kernel_normalize(self.k2d_ds, self.down_kernel)
				k2d_mean = tf.reduce_mean(self.k2d_ds, axis=[1, 2], keepdims=True)
				self.kernel_loss = l1_loss(k2d_mean, get_1d_kernel(self.ds_kernel, self.batch_size))
				self.total_loss = self.rec_loss_ds_hr + self.kernel_loss
				# validation
				self.val_rec_loss_ds_hr = l1_loss(self.val_lr, self.output_ds_hr_val)
				self.k2d_ds_val = kernel_normalize(self.k2d_ds_val, self.down_kernel)
				k2d_mean_val = tf.reduce_mean(self.k2d_ds_val, axis=[1, 2], keepdims=True)
				self.val_kernel_loss = l1_loss(k2d_mean_val, get_1d_kernel(self.ds_kernel_val, self.val_batch_size))
				self.val_total_loss = self.val_rec_loss_ds_hr + self.val_kernel_loss
				self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_lr + 1) / 2, (self.output_ds_hr_val + 1) / 2, max_val=1.0))

			elif self.training_stage == 2:
				# training
				self.rec_loss = l1_loss(self.train_hr, self.output)
				self.total_loss = self.rec_loss
				# validation
				self.val_rec_loss = l1_loss(self.val_hr, self.output_val)
				self.val_total_loss = self.val_rec_loss
				self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_hr + 1) / 2, (self.output_val + 1) / 2, max_val=1.0))

			elif self.training_stage == 3:
				# training
				self.rec_loss = l1_loss(self.train_hr, self.output)
				self.rec_loss_ds_hr = l1_loss(self.train_lr, self.output_ds_hr)
				self.k2d_ds = kernel_normalize(self.k2d_ds, self.down_kernel)
				k2d_mean = tf.reduce_mean(self.k2d_ds, axis=[1, 2], keepdims=True)
				self.kernel_loss = l1_loss(k2d_mean, get_1d_kernel(self.ds_kernel, self.batch_size))
				self.total_loss = self.rec_loss + self.rec_loss_ds_hr + self.kernel_loss
				# validation
				self.val_rec_loss = l1_loss(self.val_hr, self.output_val)
				self.val_rec_loss_ds_hr = l1_loss(self.val_lr, self.output_ds_hr_val)
				self.k2d_ds_val = kernel_normalize(self.k2d_ds_val, self.down_kernel)
				k2d_mean_val = tf.reduce_mean(self.k2d_ds_val, axis=[1, 2], keepdims=True)
				self.val_kernel_loss = l1_loss(k2d_mean_val, get_1d_kernel(self.ds_kernel_val, self.val_batch_size))
				self.val_total_loss = self.val_rec_loss + self.val_rec_loss_ds_hr + self.val_kernel_loss
				self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_hr + 1) / 2, (self.output_val + 1) / 2, max_val=1.0))

			""" Visualization """
			# visualization of GT degradation kernel
			self.ds_kernel_vis = tf.transpose(self.ds_kernel, (3, 1, 2, 0))  # [B, bicubic_size, bicubic_size, 1]
			kernel_min = tf.reduce_min(self.ds_kernel_vis, axis=(1, 2), keepdims=True)
			kernel_max = tf.reduce_max(self.ds_kernel_vis, axis=(1, 2), keepdims=True)
			self.scale_vis = (self.patch_size*self.factor)//self.bicubic_size
			self.ds_kernel_vis = local_conv_vis_ds(self.ds_kernel_vis, kernel_min, kernel_max, 3, self.scale_vis)

			# visualization of estimated degradation kernel
			if self.training_stage in [1, 3]:
				self.k2d_ds_vis = tf.reshape(k2d_mean, [self.batch_size, self.down_kernel, self.down_kernel, 1])  # [B, down_kernel, down_kernel, 1]
				self.k2d_ds_vis = local_conv_vis_ds(self.k2d_ds_vis, kernel_min, kernel_max, 3, self.scale_vis)

			# visualization of local filters in KOALA modules
			if self.training_stage == 3:
				self.filter_p = tf.reduce_mean(self.filter_p, axis=(1, 2))
				self.filter_p = tf.reshape(self.filter_p, [self.batch_size, 7, 7, 1])
				self.filter_p = local_conv_vis_ds(self.filter_p, None, None, 6, 10)

			""" Learning Rate Schedule """
			global_step = tf.Variable(initial_value=0, trainable=False)
			if self.lr_type == "stair_decay":
				self.lr_decay_boundary = [y * (self.updates_per_epoch) for y in self.lr_stair_decay_points]
				self.lr_decay_value = [self.lr * (self.lr_stair_decay_factor ** y) for y in range(len(self.lr_stair_decay_points) + 1)]
				self.reduced_lr = tf.train.piecewise_constant(global_step, self.lr_decay_boundary, self.lr_decay_value)
				print("lr_type: stair_decay")
			elif self.lr_type == "linear_decay":
				self.reduced_lr = tf.placeholder(tf.float32, name='learning_rate')
				print("lr_type: linear_decay")
			else:  # no decay
				self.reduced_lr = tf.convert_to_tensor(self.lr)
				print("lr_type: no decay")

			""" Optimizer """
			srnet_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="SISR")
			# print("\nTrainable Parameters:")
			# for param in srnet_params:
				# print(param.name)
			self.optimizer = tf.train.AdamOptimizer(self.reduced_lr).minimize(self.total_loss, global_step=global_step, var_list=srnet_params)

			"""" TensorBoard Summary """
			if self.tensorboard:
				# loss summary
				total_loss_sum = tf.summary.scalar("val_total_loss", self.val_total_loss)
				train_PSNR_sum = tf.summary.scalar("val_PSNR", self.val_PSNR)
				self.total_summary_loss = tf.summary.merge([total_loss_sum, train_PSNR_sum])
				# image summary
				lr_sum = tf.summary.image("LR", self.val_lr, max_outputs=self.val_batch_size)
				hr_sum = tf.summary.image("HR", self.val_hr, max_outputs=self.val_batch_size)
				# kernel summary
				self.ds_kernel_val_vis = tf.transpose(self.ds_kernel_val, [3, 1, 2, 0])  # [B, bicubic_size, bicubic_size, 1]
				self.ds_kernel_val_vis = local_conv_vis_ds(self.ds_kernel_val_vis, None, None, 3, self.scale_vis)
				ds_kernel_sum = tf.summary.image("Degradation Kernel (GT)", self.ds_kernel_val_vis, max_outputs=self.val_batch_size)
				self.total_summary_img = tf.summary.merge([ds_kernel_sum, lr_sum, hr_sum])
				# result summary
				if self.training_stage in [1, 3]:
					self.k2d_ds_val_vis = tf.reshape(k2d_mean_val, [self.val_batch_size, self.down_kernel, self.down_kernel, 1])
					self.k2d_ds_val_vis = local_conv_vis_ds(self.k2d_ds_val_vis, None, None, 3, self.scale_vis)
					k2d_ds_sum = tf.summary.image("Degradation Kernel (Predicted)", self.k2d_ds_val_vis, max_outputs=self.val_batch_size)
					output_sum_ds_hr = tf.summary.image("LR (Predicted)", self.output_ds_hr_val, max_outputs=self.val_batch_size)
					self.total_summary_img = tf.summary.merge([self.total_summary_img, k2d_ds_sum, output_sum_ds_hr])
				if self.training_stage in [2, 3]:
					output_sum = tf.summary.image("SR (Predicted)", self.output_val, max_outputs=self.val_batch_size)
					self.total_summary_img = tf.summary.merge([self.total_summary_img, output_sum])
		
		elif self.phase == 'test':
			assert self.training_stage == 3, "training_stage should be 3"

			""" Directories """
			self.test_img_dir = os.path.join(self.result_dir, 'imgs_test', self.model_dir)
			check_folder(self.test_img_dir)

			""" Set Data Paths """
			self.list_test_lr = data.list_test_lr  # test_data_path (LR)
			if self.eval:
				self.list_test_hr = data.list_test_hr  # test_label_path (HR)

			""" Set Placeholders """
			self.test_lr = tf.placeholder(tf.float32, (1, None, None, self.channels))
			self.test_hr = tf.placeholder(tf.float32, (1, None, None, self.channels))

			""" Define Model """
			self.k2d_ds_test = self.downsampling_network(self.test_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
			self.output_test, _ = self.upsampling_network(self.test_lr, self.k2d_ds_test, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')

		self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

	def train(self, args):
		if self.tensorboard:
			self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		saver = tf.train.Saver(max_to_keep=3)
		""" Restore Checkpoint """
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# print("####################### print tensors from checkpoints########")
			# print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path,'',False,True)
			saver.restore(self.sess, ckpt.model_checkpoint_path)
			start_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
			print("!!!!!!!!!!!!!! Restored iteration : {}".format(start_epoch))
		else:
			print("!!!!!!!!!!!!!! Learning from scratch")
			start_epoch = 1

			# load pre-trained model for downsampling_network and upsampling_network_baseline
			if self.training_stage == 3:
				print(" [*] Loading pre-trained downsampling_network model...")
				saver_ds = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SISR_DDF'))
				ckpt_ds = tf.train.get_checkpoint_state('./ckpt/downsampling_network_x{}'.format(self.factor))
				assert ckpt_ds is not None, " [!] No pretrained downsampling network - stage 1 training is needed!"
				if ckpt_ds.model_checkpoint_path:
					saver_ds.restore(self.sess, ckpt_ds.model_checkpoint_path)
					print(" [*] Restored {}".format(ckpt_ds.model_checkpoint_path))

				print(" [*] Loading pre-trained upsampling_network_baseline model...")
				saver_us = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SISR_DUF/conv2d') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SISR_DUF/Residual_block0'))
				ckpt_us = tf.train.get_checkpoint_state('./ckpt/upsampling_network_baseline_x{}'.format(self.factor))
				assert ckpt_us is not None, " [!] No pretrained upsampling network - stage 2 training is needed!"
				if ckpt_us.model_checkpoint_path:
					saver_us.restore(self.sess, ckpt_us.model_checkpoint_path)
					print(" [*] Restored {}".format(ckpt_us.model_checkpoint_path))

			# write logs
			with open(self.log_dir+'/'+self.model_dir+'.txt', 'a') as log:
				log.write('----- Model parameters -----\n')
				log.write('[{:s}] \n'.format((str(datetime.now())[:-7])))
				for arg in vars(args):
					log.write('{} : {}\n'.format(arg, getattr(args, arg)))
				log.write('\n\nepoch\tl1_loss\tPSNR\n')

		reduced_lr = self.lr
		feed_dict = {}
		for epoch in range(start_epoch, self.max_epoch+1):
			if self.lr_type == 'linear_decay':
				if epoch > self.lr_linear_decay_point:
					reduced_lr = self.lr * (1 - (epoch-float(self.lr_linear_decay_point))/(self.max_epoch-float(self.lr_linear_decay_point)))
				feed_dict = {self.reduced_lr: reduced_lr}

			""" Training """
			rec_loss = 0.0
			for i in range(self.updates_per_epoch):
				if self.training_stage == 1:
					rec_loss_temp, _, lr_per_epoch = self.sess.run([self.rec_loss_ds_hr, self.optimizer, self.reduced_lr], feed_dict)
				elif self.training_stage in [2, 3]:
					rec_loss_temp, _, lr_per_epoch = self.sess.run([self.rec_loss, self.optimizer, self.reduced_lr], feed_dict)
				rec_loss += rec_loss_temp
			print('{:s}\t\tEpoch: [{}/{}], lr : {:.8}'.format((str(datetime.now())[:-7]), epoch, self.max_epoch, lr_per_epoch))

			""" Validation """
			val_hr_batch = np.empty((self.val_batch_size, self.factor*self.val_patch_size, self.factor*self.val_patch_size, self.channels))
			val_bicubic_k = np.expand_dims(np.expand_dims(get_bicubic_kernel(self.bicubic_size, self.anti_aliasing, self.factor), axis=0), axis=3)
			val_bicubic_k = np.tile(val_bicubic_k, (1, 1, 1, self.val_batch_size))
			val_gaussian_k = np.empty((self.val_batch_size, self.gaussian_size, self.gaussian_size, 1))
			if epoch % 5 == 0:
				self.generate_sampled_image(int(epoch))
				val_cnt = 0
				psnr = 0.0
				num_val = 20
				while val_cnt < num_val:
					for b in range(self.val_batch_size):
						val_gaussian_k[b, :, :, 0] = random_anisotropic_gaussian_kernel_seed(val_cnt+b, self.gaussian_size)
						val_hr = read_img_trim(self.list_val[val_cnt + b], factor=self.factor)
						# crop index
						_, h, w, _ = val_hr.shape
						h_idx = int(np.floor(h / 2) - np.floor(self.val_patch_size / 2 * self.factor))
						w_idx = int(np.floor(w / 2) - np.floor(self.val_patch_size / 2 * self.factor))
						# crop center
						val_hr = val_hr[:, h_idx:h_idx + self.factor * self.val_patch_size, w_idx:w_idx + self.factor * self.val_patch_size, :]
						# store as batch
						val_hr_batch[b, :, :, :] = val_hr

					if self.tensorboard:
						if (val_cnt == 0) & (epoch % 50 == 0):  # add summary_img
							summary_loss, summary_img, val_PSNR = self.sess.run([self.total_summary_loss, self.total_summary_img, self.val_PSNR],
																				{self.val_base_k: val_bicubic_k, self.val_rand_k: val_gaussian_k, self.val_hr: val_hr_batch})
							self.writer.add_summary(summary_loss, epoch)
							self.writer.add_summary(summary_img, epoch)
						else:  # only summary_loss (for speed)
							summary_loss, val_PSNR = self.sess.run([self.total_summary_loss, self.val_PSNR],
																   {self.val_base_k: val_bicubic_k, self.val_rand_k: val_gaussian_k, self.val_hr: val_hr_batch})
							self.writer.add_summary(summary_loss, epoch)
					else:
						val_PSNR = self.sess.run(self.val_PSNR, {self.val_base_k: val_bicubic_k, self.val_rand_k: val_gaussian_k, self.val_hr: val_hr_batch})
					psnr += val_PSNR
					val_cnt += self.val_batch_size
				psnr /= (num_val / self.val_batch_size)
				rec_loss = rec_loss / (self.updates_per_epoch)
				print('Validation: Recon loss {:.8}, PSNR {:.4} dB'.format(rec_loss, psnr))
				with open(self.log_dir+'/'+self.model_dir+'.txt', 'a') as log:
					log.write('{}\t{:.4}\t{:.4}\n'.format(epoch, rec_loss, psnr))

			# save network weights
			if epoch % 10 == 0:
				print('Saving the model...')
				saver.save(self.sess, os.path.join(self.ckpt_dir, self.model_name), epoch)

	def test(self):
		assert self.training_stage == 3, "training_stage should be 3"
		# saver to save model
		self.saver = tf.train.Saver()
		# restore checkpoint
		ckpt = tf.train.get_checkpoint_state(os.path.join(self.test_ckpt_path, self.model_dir))
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print("!!!!!!!!!!!!!! Restored from {}".format(ckpt.model_checkpoint_path))

		"""" Test """
		avg_inf_time = 0.0
		avg_test_PSNR = 0.0
		patch_boundary = 0
		for test_cnt in range(len(self.list_test_lr)):
			test_lr = read_img_trim(self.list_test_lr[test_cnt], factor=4*self.test_patch[0])
			test_lr = check_gray(test_lr)
			if self.eval:
				test_hr = read_img_trim(self.list_test_hr[test_cnt], factor=self.factor*4*self.test_patch[0])
				test_hr = check_gray(test_hr)
			_, h, w, c = test_lr.shape
			output_test = np.zeros((1, h*self.factor, w*self.factor, c))
			inf_time = 0.0
			# test image divided into test_patch[0]*test_patch[1] to fit memory (default: 1x1)
			for p in range(self.test_patch[0] * self.test_patch[1]):
				pH = p // self.test_patch[1]
				pW = p % self.test_patch[1]
				sH = h // self.test_patch[0]
				sW = w // self.test_patch[1]
				# process data considering patch boundary
				H_low_ind, H_high_ind, W_low_ind, W_high_ind = get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW)
				test_lr_p = test_lr[:, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :]
				st = time.time()
				output_test_p = self.sess.run([self.output_test], feed_dict={self.test_lr: test_lr_p})
				inf_time_p = time.time() - st
				inf_time += inf_time_p
				output_test_p = trim_patch_boundary(output_test_p, patch_boundary, h, w, pH, sH, pW, sW, self.factor)
				output_test[:, pH * sH * self.factor: (pH + 1) * sH * self.factor, pW * sW * self.factor: (pW + 1) * sW * self.factor, :] = output_test_p
			avg_inf_time += inf_time
			# compute PSNR and print results
			if self.eval:
				test_PSNR = compute_y_psnr(output_test, test_hr)
				avg_test_PSNR += test_PSNR
				print(" <Test> [%4d/%4d]-th images, time: %4.4f(seconds), test_PSNR: %2.2f[dB]  "
					  % (int(test_cnt+1), len(self.list_test_lr), inf_time, test_PSNR))
			else:
				print(" <Test> [%4d/%4d]-th images, time: %4.4f(seconds)  "
					  % (int(test_cnt + 1), len(self.list_test_lr), inf_time))
			# save predicted SR images
			save_path = os.path.join(self.test_img_dir, os.path.basename(self.list_test_lr[test_cnt]))
			save_img(output_test, save_path)

		if self.eval:
			avg_test_PSNR /= float(len(self.list_test_lr))
			print("######### Average Test PSNR: %.8f[dB]  #########" % avg_test_PSNR)
		avg_inf_time /= float(len(self.list_test_lr))
		print("######### Average Inference Time: %.8f[s]  #########" % avg_inf_time)


	@property
	def model_dir(self):
		return "{}_x{}".format(self.model_name, self.factor)

	def generate_sampled_image(self, epoch):
		patch_size = self.patch_size
		if self.training_stage == 1:
			grid = patch_size
		else:
			grid = int(patch_size*self.factor)

		n = min(self.n_display, self.batch_size)
		if self.training_stage == 1:
			train_lr, output_ds_hr, ds_kernel_vis, k2d_ds_vis = self.sess.run([self.train_lr, self.output_ds_hr, self.ds_kernel_vis, self.k2d_ds_vis])
			combined_img = np.zeros((n*grid, 4*grid, 3))
		elif self.training_stage == 2:
			train_lr, train_hr, output, ds_kernel_vis = self.sess.run([self.train_lr, self.train_hr, self.output, self.ds_kernel_vis])
			combined_img = np.zeros((n*grid, 4*grid, 3))
		elif self.training_stage == 3:
			train_lr, train_hr, output, ds_kernel_vis, k2d_ds_vis, filter_p = self.sess.run([self.train_lr, self.train_hr, self.output, self.ds_kernel_vis, self.k2d_ds_vis, self.filter_p])
			combined_img = np.zeros((n*grid, 6*grid, 3))

		for i in range(0,n):
			if self.training_stage == 1:
				combined_img[i*grid:(i+1)*grid, 0*grid:1*grid] = imresize(ds_kernel_vis[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 1*grid:2*grid] = imresize(k2d_ds_vis[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 2*grid:3*grid] = train_lr[i, :]
				combined_img[i*grid:(i+1)*grid, 3*grid:4*grid] = output_ds_hr[i, :]
			elif self.training_stage == 2:
				combined_img[i*grid:(i+1)*grid, 0*grid:1*grid] = imresize(ds_kernel_vis[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 1*grid:2*grid] = imresize(train_lr[i, :], self.factor)
				combined_img[i*grid:(i+1)*grid, 2*grid:3*grid] = output[i, :]
				combined_img[i*grid:(i+1)*grid, 3*grid:4*grid] = train_hr[i, :]
			elif self.training_stage == 3:
				combined_img[i*grid:(i+1)*grid, 0*grid:1*grid] = imresize(ds_kernel_vis[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 1*grid:2*grid] = imresize(k2d_ds_vis[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 2*grid:3*grid] = imresize(filter_p[i, :], output_shape=(grid,grid))
				combined_img[i*grid:(i+1)*grid, 3*grid:4*grid] = imresize(train_lr[i, :], self.factor)
				combined_img[i*grid:(i+1)*grid, 4*grid:5*grid] = output[i, :]
				combined_img[i*grid:(i+1)*grid, 5*grid:6*grid] = train_hr[i, :]
			
		combined_img = np.clip(combined_img, -1.0, 1.0)

		combined_img = Image.fromarray(((np.squeeze(combined_img) + 1.0) / 2.0 * 255).astype(np.uint8))
		combined_img.save(os.path.join(self.img_dir, 'img_'+'{:05d}'.format(epoch)+'.jpg'))
		print("!!!!!!!!!!!   Output image saved  !!!!!!!!!!!! (check ./{})".format(self.img_dir))
