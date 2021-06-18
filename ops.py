import tensorflow as tf


##################################################################################
# Network Blocks
##################################################################################

def res_block(x, out_ch, k_sz, scope='Residual_block'):
	# residual block
	with tf.variable_scope(scope):
		n = conv2d(tf.nn.relu(x), out_ch, k_sz)
		n = conv2d(tf.nn.relu(n), out_ch, k_sz)
	return x + n


def enc_level_res(x, out_ch, pool_factor=2, scope='enc_block_res'):
	# encoder level with resblocks
	with tf.variable_scope(scope):
		n = conv2d(x, out_ch, 3)
		n = res_block(n, out_ch, 3, 'res_block/0')
		n = tf.nn.relu(res_block(n, out_ch, 3, 'res_block/1'))
		skip = n
		n = max_pool(n, pool_factor)
	return n, skip


def bottleneck_res(x, out_ch, scope='bottleneck_res'):
	# bottleneck using resblock
	with tf.variable_scope(scope):
		n = conv2d(x, out_ch, 3)
		n = tf.nn.relu(res_block(n, out_ch, 3))
	return n


def dec_level_res(x, skip, out_ch, stride=2, scope='dec_block_res'):
	# decoder level with resblocks
	with tf.variable_scope(scope):
		n = deconv2d(x, out_ch, 4, stride)
		n = tf.concat([n, skip], 3)
		n = conv2d(n, out_ch, 3)
		n = res_block(n, out_ch, 3, 'res_block/0')
		n = tf.nn.relu(res_block(n, out_ch, 3, 'res_block/1'))
	return n


def koala(x, kernel, feat_ch, ker_ch, conv_k_sz, lc_k_sz, scope_res='Residual_block0_', scope='KOALA_module'):
	# kernel-oriented adaptive local adjustment (KOALA) module
	with tf.variable_scope(scope_res):
		n = conv2d(tf.nn.relu(x), feat_ch, conv_k_sz)
		n = conv2d(tf.nn.relu(n), feat_ch, conv_k_sz)
	with tf.variable_scope(scope):
		# multiplicative parameters
		mul_p = conv2d(kernel, feat_ch, conv_k_sz)
		mul_p = conv2d(tf.nn.relu(mul_p), feat_ch, conv_k_sz)
		# local filtering parameters
		filter_p = conv2d(kernel, ker_ch, k_sz=1)  # 1x1 conv
		filter_p = conv2d(tf.nn.relu(filter_p), lc_k_sz*lc_k_sz, k_sz=1)  # 1x1 conv
		# spatially-variant feature filtering
		n = tf.multiply(n, mul_p)
		n = local_conv_feat(n, filter_p, feat_ch, lc_k_sz)
	n = x+n
	return n, filter_p


def cr_block(x, num_blocks, out_ch, k_sz, scope='condition'):
	# conv-relu stack
	with tf.variable_scope(scope):
		n = tf.nn.relu(conv2d(x, out_ch, k_sz))
		for i in range(num_blocks-1):
			n = tf.nn.relu(conv2d(n, out_ch, k_sz))
	return n


##################################################################################
# Layers
##################################################################################

def conv2d(x, out_ch, k_sz, stride=1):
	# convolution layer
	init = tf.contrib.layers.xavier_initializer(uniform=False)
	n = tf.layers.conv2d(x, out_ch, k_sz, stride, 'same', kernel_initializer=init)
	return n


def deconv2d(x, out_ch, k_sz=4, stride=2):
	# deconvolution layer
	init = tf.contrib.layers.xavier_initializer(uniform=False)
	n = tf.layers.conv2d_transpose(x, out_ch, k_sz, stride, 'same', kernel_initializer=init)
	return n


def max_pool(x, pool_factor):
	# max pooling layer
	n = tf.nn.max_pool(x, [1, pool_factor, pool_factor, 1], [1, pool_factor, pool_factor, 1], 'SAME')
	return n


def local_conv_us(img, kernel_2d, factor, num_ch, k_sz):
	# local filtering operation for upsampling network
	# img: [B, H, W, num_ch]
	# kernel_2d: [B, H, W, k_sz*k_sz*factor*factor]

	# [B, H, W, k*k*c]
	img = tf.image.extract_image_patches(img, ksizes=(1, k_sz, k_sz, 1), strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding="SAME")
	img = tf.split(img, k_sz*k_sz, axis=-1)  # k*k of [B, H, W, c]
	img = tf.stack(img, axis=3)  # [B, H, W, k*k, c]
	img = tf.tile(img, [1, 1, 1, 1, factor*factor])  # [B, H, W, k*k, f*f*c]

	kernel_2d = tf.split(kernel_2d, k_sz*k_sz, axis=-1)  # k*k of [B, H, W, f*f]
	kernel_2d = tf.stack(kernel_2d, axis=3)  # [B, H, W, k*k, f*f]
	kernel_2d = kernel_normalize(kernel_2d, k_sz)
	
	kernel_2d = tf.expand_dims(kernel_2d, -1)  # [B, H, W, k*k, f*f, 1]
	kernel_2d = tf.tile(kernel_2d, [1, 1, 1, 1, 1, num_ch])  # [B, H, W, k*k, f*f, c]
	kernel_2d = tf.unstack(kernel_2d, axis=4)  # f*f of [B, H, W, k*k, c]
	kernel_2d = tf.concat(kernel_2d, axis=4)  # [B, H, W, k*k, f*f*c]

	result = tf.multiply(img, kernel_2d)  # element-wise multiplication, resulting in [B, H, W, k*k, f*f*c]
	result = tf.reduce_sum(result, axis=3)  # [B, H, W, f*f*c]
	result = tf.depth_to_space(result, factor)  # [B, f*H, f*W, c]

	return result


def local_conv_ds(img, kernel_2d, factor, num_ch, k_sz):
	# local filtering operation for downsampling network
	# img: [B, H, W, num_ch]
	# kernel_2d: [B, H, W, kernel*kernel]

	# [B, H, W, k*k*c]
	img = tf.image.extract_image_patches(img, ksizes=(1, k_sz, k_sz, 1), strides=(1, factor, factor, 1), rates=(1, 1, 1, 1), padding="SAME")
	img = tf.split(img, k_sz * k_sz, axis=-1)  # k*k of [B, H, W, c]
	img = tf.stack(img, axis=3)  # [B, H, W, k*k, c]

	kernel_2d = kernel_normalize(kernel_2d, k_sz)
	kernel_2d = tf.expand_dims(kernel_2d, -1)  # [B, H, W, k*k, 1]
	kernel_2d = tf.tile(kernel_2d, [1, 1, 1, 1, num_ch])  # [B, H, W, k*k, c]

	result = tf.multiply(img, kernel_2d)  # element-wise multiplication, resulting in [B, H, W, k*k, c]
	result = tf.reduce_sum(result, axis=3)  # [B, H, W, c]

	return result


def local_conv_feat(img, kernel_2d, num_ch, k_sz):
	# local filtering operation for features
	# img: [B, H, W, num_ch]
	# kernel_2d: [B, H, W, kernel*kernel]

	# [B, H, W, k*k*c]
	img = tf.image.extract_image_patches(img, ksizes=(1, k_sz, k_sz, 1), strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding="SAME")
	img = tf.split(img, k_sz * k_sz, axis=-1)  # k*k of [B, H, W, c]
	img = tf.stack(img, axis=3)  # [B, H, W, k*k, c]

	kernel_2d = kernel_normalize(kernel_2d, k_sz)
	kernel_2d = tf.expand_dims(kernel_2d, -1)  # [B, H, W, k*k, 1]
	kernel_2d = tf.tile(kernel_2d, [1, 1, 1, 1, num_ch])  # [B, H, W, k*k, c]

	result = tf.multiply(img, kernel_2d)  # element-wise multiplication, resulting in [B, H, W, k*k, c]
	result = tf.reduce_sum(result, axis=3)  # [B, H, W, c]

	return result


def kernel_normalize(kernel_2d, k_sz):
	kernel_2d = kernel_2d - tf.reduce_mean(kernel_2d, axis=3, keepdims=True)
	kernel_2d = kernel_2d + 1.0 / (k_sz ** 2)
	return kernel_2d


##################################################################################
# Loss Function
##################################################################################

def l1_loss(x, y):
	loss = tf.reduce_mean(tf.abs(x - y))
	return loss


##################################################################################
# Degradation
##################################################################################

def get_ds_kernel(base_kernel, rand_kernel):
	# convolve base_kernel with rand_kernel
	# base kernel: bicubic, random kernel: anisotropic gaussian
	rand_kernel = tf.transpose(rand_kernel, [1, 2, 0, 3])  # [gaussian_size, gaussian_size, B, 1]
	ds_kernel = tf.nn.depthwise_conv2d(base_kernel, filter=rand_kernel, strides=[1, 1, 1, 1], padding='SAME')  # [1, bicubic_size, bicubic_size, B]
	return ds_kernel


def get_ds_input(hr, ds_kernel, num_ch, batch_size, pad_l, pad_r, factor):
	# convolve HR image with the downsampling kernel to obtain input LR
	ds_kernel = tf.squeeze(ds_kernel, 0)  # [bicubic_size, bicubic_size, B]
	ds_kernel = tf.expand_dims(ds_kernel, 3)  # [bicubic_size, bicubic_size, B, 1]
	ds_kernel = tf.tile(ds_kernel, [1, 1, 1, num_ch])  # [bicubic_size, bicubic_size, B, channels]
	ds_kernel = tf.unstack(ds_kernel, batch_size, axis=2)  # B*[bicubic_size, bicubic_size, channels]
	ds_kernel = tf.concat(ds_kernel, axis=2)  # [bicubic_size, bicubic_size, B*channels]
	ds_kernel = tf.expand_dims(ds_kernel, 3)  # [bicubic_size, bicubic_size, B*channels, 1]

	lr = tf.unstack(hr, batch_size, axis=0)  # B*[H, W, C]
	lr = tf.concat(lr, axis=2)  # [H, W, B*C]
	lr = tf.expand_dims(lr, 0)  # [1, H, W, B*C]
	lr = tf.pad(lr, [[0, 0], [pad_l, pad_r], [pad_l, pad_r], [0, 0]], 'symmetric')
	lr = tf.nn.depthwise_conv2d(lr, filter=ds_kernel, strides=[1, factor, factor, 1], padding='VALID')  # [1, H, W, B*C]
	lr = tf.split(lr, batch_size, axis=3)  # B*[1, H, W, C]
	lr = tf.concat(lr, axis=0)  # [B, H, W, C]
	return lr


def get_1d_kernel(flattened_kernel, batch_size):
	# flattened_kernel : [1, k_sz, k_sz, B]
	# kernel_1d : [B, 1, 1, k_sz*k_sz]
	kernel_1d = tf.transpose(flattened_kernel, [3,1,2,0]) # [B, k_sz, k_sz, 1]
	kernel_1d = tf.reshape(kernel_1d, [batch_size,1,1,-1]) # [B, 1, 1, k_sz*k_sz]
	return kernel_1d


##################################################################################
# Visualization
##################################################################################

def local_conv_vis_ds(kernel_2d, kernel_min=None, kernel_max=None, padding=0, scale=1):
	if kernel_min is None:
		kernel_min = tf.reduce_min(kernel_2d, axis=(1, 2), keepdims=True)
	if kernel_max is None:
		kernel_max = tf.reduce_max(kernel_2d, axis=(1, 2), keepdims=True)
	if padding != 0:
		kernel_2d = tf.pad(kernel_2d, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'constant')
	kernel_2d = 2.0*(kernel_2d-kernel_min)/(kernel_max-kernel_min)-1.0
	if scale != 1:
		kernel_2d = nearest_neighbor(kernel_2d, scale)
	kernel_2d = tf.concat((kernel_2d, kernel_2d, kernel_2d), axis=3)
	return kernel_2d


def nearest_neighbor(x, factor):
	y = tf.tile(x, [1, 1, 1, factor*factor])
	y = tf.depth_to_space(y, factor)
	return y
