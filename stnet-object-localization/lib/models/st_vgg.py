from utils.config import cfg
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import selective_tuning as st
from utils.miscellaneous import calculate_net_specs

__all__ = [
	'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
	'vgg19_bn', 'vgg19',
]


model_urls = {
	'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
	'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
	'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
	'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
	net_params_ini = [
		['d', {'kernel': 1, 'stride': 1, 'padding': 0}],

	]
	net_params_mid = [
		['c', {'kernel': 6, 'stride': 1, 'padding': 0}]
	]
	net_params_class = [
		['l', {'kernel': 1, 'stride': 1, 'padding': 0}],
		['l', {'kernel': 1, 'stride': 1, 'padding': 0}],
	]

	#cfg.VALID.INPUT_SIZE[-1]
	def __init__(self, features, num_classes=1000, init_weights=True):
		super(VGG, self).__init__()
		self.g_top = None
		self.net_params = self.net_params_ini + features[1] + self.net_params_mid + self.net_params_class
		#print(self.net_params)

		self.net_specs = calculate_net_specs(self.net_params, cfg.VALID.INPUT_SIZE[-1], verbose=True)
		#features
		self.a_conv_01 = st.AttentiveConv(3, 64, kernel_size=3, padding=1)
		self.a_pool_02 =st.AttentivePool(kernel_size=2, stride=2)
		self.a_conv_03 = st.AttentiveConv(64, 128, kernel_size=3, padding=1)
		self.a_pool_04 = st.AttentivePool(kernel_size=2, stride=2)
		self.a_conv_05 = st.AttentiveConv(128, 256, kernel_size=3, padding=1)
		self.a_conv_06 = st.AttentiveConv(256, 256, kernel_size=3, padding=1)
		self.a_pool_07 = st.AttentivePool(kernel_size=2, stride=2)
		self.a_conv_08 = st.AttentiveConv(256, 512, kernel_size=3, padding=1)
		self.a_conv_09 = st.AttentiveConv(512, 512, kernel_size=3, padding=1)
		self.a_pool_10 = st.AttentivePool(kernel_size=2, stride=2)
		self.a_conv_11 = st.AttentiveConv(512, 512, kernel_size=3, padding=1)
		self.a_conv_12 = st.AttentiveConv(512, 512, kernel_size=3, padding=1)
		self.a_pool_13 = st.AttentivePool(kernel_size=2, stride=2)
		#classification

		self.a_conv14 = st.AttentiveConv(512, 4096, kernel_size=7, padding=0)
		self.a_drop14 = nn.Dropout()
		self.bridge15 = st.AttentiveBridge(4096, cfg.ST.LINEAR_B_MODE, cfg.ST.LINEAR_B_OFFSET)  # 512 * 7 * 7
		self.a_linear_16 = st.AttentiveLinear(4096, 4096, cfg.ST.LINEAR_S_MODE, cfg.ST.LINEAR_S_OFFSET)
		self.a_drop16 = nn.Dropout()
		self.a_linear_17 = st.AttentiveLinear(4096, num_classes, cfg.ST.LINEAR_S_MODE, cfg.ST.LINEAR_S_OFFSET)

		if init_weights:
			self._initialize_weights()

	def attend(self, g):

		self.g_top = g

		g = self.a_linear_17.attend(g)

		g = self.a_linear_16.attend(g)


		g = self.bridge15.attend(self.g_top, g)
		g = g.view(len(g), 4096, 1, 1)
		g = self.a_conv14.attend(g)
		
		g = self.a_pool_13.attend(g)
		g = self.a_conv_12.attend(g)
		g = self.a_conv_11.attend(g)
		g = self.a_pool_10.attend(g)
		g = self.a_conv_09.attend(g)
		g = self.a_conv_08.attend(g)
		g = self.a_pool_07.attend(g)
		g = self.a_conv_06.attend(g)
		g = self.a_conv_05.attend(g)
		g = self.a_pool_04.attend(g)
		g = self.a_conv_03.attend(g)
		if cfg.ST.BOTTOM == 2:
			return g
		g = self.a_pool_02.attend(g)
		g = self.a_conv_01.attend(g)

		return g



	def forward(self, x):

		x = self.a_conv_01(x)
		x = self.a_pool_02(x)
		x = self.a_conv_03(x)
		x = self.a_pool_04(x)
		x = self.a_conv_05(x)
		x = self.a_conv_06(x)
		x = self.a_pool_07(x)
		x = self.a_conv_08(x)
		x = self.a_conv_09(x)
		x = self.a_pool_10(x)
		x = self.a_conv_11(x)
		x = self.a_conv_12(x)
		x = self.a_pool_13(x)
				# classification

		#x = x.view(x.size(0), 256 * 7 * 7)
		x = x.view(x.size(0), 512, 7, 7)
		x = self.a_conv14(x)
		x = x.view(x.size(0), 4096)

		x = self.a_drop14(x)
		#x = self.bridge15(x)
		x = self.a_linear_16(x)


		x = x.view(x.size(0), 4096)
		x = self.a_drop16(x)
		x = self.a_linear_17(x)

		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_state_dict(self, state_dict_pt, strict=True):
		skip_params = []
		state_dict_self = self.state_dict()

		state_dict_self_keys = iter(state_dict_self.keys())

		for i, (name, param) in enumerate(state_dict_pt.items()):

			if name in skip_params:
				continue
			state_dict_self_key = next(state_dict_self_keys)

			if name == 'classifier.0.weight':
				param = param.view(state_dict_self[state_dict_self_key].shape)

			while not state_dict_self[state_dict_self_key].shape == param.shape:
				state_dict_self_key = next(state_dict_self_keys)
			state_dict_self[state_dict_self_key].copy_(param)



def make_layers(l_cfg, batch_norm=False, mode='forward'):
	in_channels = 3 #image  rgb = 3
	if mode != 'forward':
		cfg = reversed(cfg)
		in_channels = 512 #4096
	layers = []

	for v in l_cfg:
		if v == 'M':
			layers += [st.AttentivePool(kernel_size=2, stride=2)]
		else:
			conv2d = st.AttentiveConv(in_channels, v, kernel_size=3, padding=1) #it has relu
			if batch_norm: #not needed
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] #ask relu and batchnorm
			else:
				layers += [conv2d] #[conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


l_cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers2(l_cfg, batch_norm=False):
	layers = []
	net_params = []
	in_channels = 3
	for v in l_cfg:
		if v == 'M':
			layers += [st.AttentivePool(kernel_size=2, stride=2)]
			net_params += [['p', {'kernel': 2, 'stride': 2, 'padding': 0}]]
		else:
			conv2d = st.AttentiveConv(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d] #[conv2d, nn.ReLU(inplace=True)]
				net_params += [['c', {'kernel': 3, 'stride': 1, 'padding': 1}]]
			in_channels = v
	return nn.Sequential(*layers), net_params

def vgg11(pretrained=True, **kwargs):
	"""VGG 11-layer model (configuration "A")
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(l_cfg['A']), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
	return model


def vgg11_bn(pretrained=True, **kwargs):
	"""VGG 11-layer model (configuration "A") with batch normalization
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['A'], batch_norm=True), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
	return model


def vgg13(pretrained=True, **kwargs):
	"""VGG 13-layer model (configuration "B")
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['B']), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
	return model


def vgg13_bn(pretrained=True, **kwargs):
	"""VGG 13-layer model (configuration "B") with batch normalization
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['B'], batch_norm=True), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
	return model


def vgg16(pretrained=True, **kwargs):
	"""VGG 16-layer model (configuration "D")
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['D']), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
	return model


def vgg16_bn(pretrained=True, **kwargs):
	"""VGG 16-layer model (configuration "D") with batch normalization
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['D'], batch_norm=True), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
	return model


def vgg19(pretrained=True, **kwargs):
	"""VGG 19-layer model (configuration "E")
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['E']), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
	return model


def vgg19_bn(pretrained=True, **kwargs):
	"""VGG 19-layer model (configuration 'E') with batch normalization
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		kwargs['init_weights'] = False
	model = VGG(make_layers2(cfg['E'], batch_norm=True), **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
	return model
