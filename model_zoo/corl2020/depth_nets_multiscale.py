
from typing_extensions import Self
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torchvision.models as models



class s_conv(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, stride, filter_size, padding=None):
        super(s_conv, self).__init__()
        if padding == None:
            padding = filter_size//2 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        
        self.conv = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data.normal_(std= 2/ self.filter_size / self.filter_size / self.out_channel)


    def forward(self, x):
        return self.conv(x)
    

class _bn(torch.nn.Module):
    
    def __init__(self, num_features):
        super(_bn, self).__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=num_features)
        
    def forward(self, x):
        return self.bn(x)
    
class s_relu(torch.nn.Module):
    
    def __init__(self):
        super(s_relu, self).__init__()
    
    def forward(self, x):
        return torch.nn.functional.relu(x)
    

class s_residual_block_first(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, stride, normalizer_fn=None):
        super(s_residual_block_first, self).__init__()
        normalizer_fn = normalizer_fn or _bn
        if in_channel == out_channel:
            if stride == 1:
                self.shortcut = torch.nn.Identity()
            else:
                self.shortcut = torch.nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0)
        else:
            self.shortcut = s_conv(in_channel=in_channel, out_channel=out_channel, filter_size=1, stride=stride, padding=0)
        # Residual
        self.conv_1 = s_conv(in_channel=in_channel, out_channel=out_channel, filter_size=3, stride=stride, padding=0)
        self.normalizer_fn1 = normalizer_fn(num_features=out_channel)
        self.relu = s_relu()
        self.conv_2 = s_conv(in_channel=out_channel, out_channel=out_channel, filter_size=3, stride=1, padding=None)
        self.normalizer_fn2 = normalizer_fn(num_features=out_channel)

    def tf_pad_stride2_1(self, x):
        return torch.nn.functional.pad(x, [0,1,0,1])

    def forward(self, x):
        # Residual
        shortcut = self.shortcut(x)
        x = self.conv_1(self.tf_pad_stride2_1(x))
        x = self.normalizer_fn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.normalizer_fn2(x)
        # Merge
        x = x + shortcut
        x = self.relu(x)
        return x
        
        

class s_residual_block(torch.nn.Module):
    
    def __init__(self, in_channel, normalizer_fn=None):
        normalizer_fn = normalizer_fn or _bn
        super(s_residual_block, self).__init__()
        self.conv_1 = s_conv(in_channel=in_channel, out_channel=in_channel, stride=1, filter_size=3)
        self.normalizer_fn1 = normalizer_fn(num_features=in_channel)
        self.relu = s_relu()
        
        self.conv_2 = s_conv(in_channel=in_channel, out_channel=in_channel, stride=1, filter_size=3)
        self.normalizer_fn2 = normalizer_fn(num_features=in_channel)
        
    def forward(self, x):
        shortcut = x
        x = self.conv_1(x)
        x = self.normalizer_fn1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.normalizer_fn2(x)
        # Merge
        x = x + shortcut
        x = self.relu(x)
        return x
    
# ONLY THE DEPTH_ENCODER USE RandomizedLayerNorm
class RandomizedLayerNorm(torch.nn.Module):
    def __init__(self, num_features, stddev=0.5):
        super(RandomizedLayerNorm, self).__init__()
        self.stddev = stddev
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        
    
    def torch_batch_norm(self, x, mean, variance, offset, scale, variance_epsilon, name=None):
        inv = torch.rsqrt(variance+variance_epsilon)
        scale = scale.repeat(inv.shape[0],1).unsqueeze(2).unsqueeze(3)
        offset = offset.repeat(inv.shape[0],1).unsqueeze(2).unsqueeze(3)
        if scale is not None:
            inv = inv * scale
        return x * inv + (offset-mean*inv if offset is not None else -mean*inv)
    
    
    def forward(self, x):
        inputs_shape = x.shape
        params_shape = inputs_shape[1]
        variance, mean = torch.var_mean(x, dim=[2,3], keepdims=True, unbiased=True)
        if self.training:
            mean_noise = torch.nn.init.trunc_normal_(torch.empty_like(mean), std=self.stddev, a=-1, b=1).detach()
            mean = mean * (1 + mean_noise)
            var_noise = torch.nn.init.trunc_normal_(torch.empty_like(variance), std=self.stddev, a=-1, b=1).detach()
            variance = variance * (1 + var_noise)
        outputs = self.torch_batch_norm(
                x,
                mean,
                variance,
                offset=self.beta,
                scale=self.gamma,
                variance_epsilon=1e-3)
        outputs.view(x.shape)
        return outputs
        

class ResNetEncoder(torch.nn.Module):
    
    def __init__(self, in_channel, normalizer_fn):
        super(ResNetEncoder, self).__init__()
        normalizer_fn = normalizer_fn or _bn
        encoder_filters = [64, 64, 128, 256, 512]
        stride = 2
        self.relu = s_relu()
        # BLOCK 1
        self.block1_conv = s_conv(in_channel=in_channel, out_channel=encoder_filters[0], stride=stride, filter_size=7, padding=0)
        self.block1_normalizer_fn = normalizer_fn(num_features=encoder_filters[0])
        self.block1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # BLOCK 2
        self.block2_residual1 = s_residual_block(in_channel=encoder_filters[0], normalizer_fn=normalizer_fn)
        self.block2_residual2 = s_residual_block(in_channel=encoder_filters[0], normalizer_fn=normalizer_fn)

        # BLOCK 3
        self.block3_residual1 = s_residual_block_first(in_channel=encoder_filters[0], out_channel=encoder_filters[2], stride=stride, normalizer_fn=normalizer_fn)
        self.block3_residual2 = s_residual_block(in_channel=encoder_filters[2], normalizer_fn=normalizer_fn)

        # BLOCK 4
        self.block4_residual1 = s_residual_block_first(in_channel=encoder_filters[2], out_channel=encoder_filters[3], stride=stride, normalizer_fn=normalizer_fn)
        self.block4_residual2 = s_residual_block(in_channel=encoder_filters[3], normalizer_fn=normalizer_fn)

        # BLOCK 5
        self.block5_residual1 = s_residual_block_first(in_channel=encoder_filters[3], out_channel=encoder_filters[4], stride=stride, normalizer_fn=normalizer_fn)
        self.block5_residual2 = s_residual_block(in_channel=encoder_filters[4], normalizer_fn=normalizer_fn)
    
    def tf_pad_stride2_2(self, x):
        return torch.nn.functional.pad(x, [2,3,2,3])

    def tf_pad_stride2_1(self, x):
        return torch.nn.functional.pad(x, [0,1,0,1])

    def forward(self, x):
        # conv1
        x = self.block1_conv(self.tf_pad_stride2_2(x))
        x = self.block1_normalizer_fn(x)
        econv1 = self.relu(x)
        x = self.block1_maxpool(self.tf_pad_stride2_1(econv1))
        # conv2
        x = self.block2_residual1(x)
        econv2 = self.block2_residual2(x)
        # conv3
        x = self.block3_residual1(econv2)
        econv3 = self.block3_residual2(x)
        # conv4
        x = self.block4_residual1(econv3)
        econv4 = self.block4_residual2(x)
        # conv5
        x = self.block5_residual1(econv4)
        econv5 = self.block5_residual2(x)
        return econv5, (econv4, econv3, econv2, econv1)

        
class layer_conv_transpose(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding_mode="constant"):
        super(layer_conv_transpose, self).__init__()
        # self.conv_transpose = torch.nn.Sequential(
            # torch.nn.ConvTranspose2d(bias=True, in_channels=in_channel, out_channels=out_channel,
            #                          kernel_size=kernel_size, stride=stride, padding=1, 
            #                          output_padding=1, padding_mode=padding_mode),
            # torch.nn.ReLU(inplace=True)
        # )

        self.conv_transpose = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(bias=True, in_channels=in_channel, out_channels=out_channel,
                                     kernel_size=kernel_size, stride=1, padding=0),
            torch.nn.ReLU(inplace=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def conv_transpose_pad(self, x):
        w = x.new_zeros(2, 2)
        w[0,0] = 1
        padded_x = torch.nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, 2, 2), stride=2, groups=x.size(1))[:,:,:-1,:-1]
        return padded_x

    def forward(self, x):
        padded_x = self.conv_transpose_pad(x)
        return self.conv_transpose(padded_x)[:,:,:-1,:-1]

class layer_conv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=None, padding_mode="constant"):
        super(layer_conv, self).__init__()
        if padding==None: padding = 0
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(bias=True, in_channels=in_channel, out_channels=out_channel, 
                            kernel_size=kernel_size, stride=stride, 
                            padding=padding, padding_mode=padding_mode),
            torch.nn.ReLU(inplace=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, x):
        return self.conv(x)
        

class DepthDecoder(torch.nn.Module):
    
    def __init__(self, encoder_bottleneck_channel, reflect_padding=False):
        super(DepthDecoder, self).__init__()
        decoder_filters = [16, 32, 64, 128, 256]
        padding_mode = 'reflect' if reflect_padding else 'zeros'
        self.padding_mode = padding_mode
        self.upconv5 = layer_conv_transpose(in_channel=encoder_bottleneck_channel, 
                                            out_channel=decoder_filters[4], kernel_size=3, 
                                            stride=2, padding_mode=padding_mode)
        self.iconv5 =  layer_conv(in_channel=decoder_filters[4]*2, 
                                 out_channel=decoder_filters[4], kernel_size=3, 
                                 stride=1, padding=0, padding_mode=padding_mode)
        
        self.upconv4 = layer_conv_transpose(in_channel=decoder_filters[4], 
                                            out_channel=decoder_filters[3], kernel_size=3, 
                                            stride=2, padding_mode=padding_mode)
        self.iconv4 = layer_conv(in_channel=decoder_filters[3]*2, 
                                 out_channel=decoder_filters[3],  kernel_size=3, 
                                 stride=1, padding=0, padding_mode=padding_mode)
        
        self.upconv3 = layer_conv_transpose(in_channel=decoder_filters[3], 
                                            out_channel=decoder_filters[2], kernel_size=3, 
                                            stride=2, padding_mode=padding_mode)
        self.iconv3 = layer_conv(in_channel=decoder_filters[2]*2, 
                                 out_channel=decoder_filters[2],  kernel_size=3, 
                                 stride=1, padding=0, padding_mode=padding_mode)
        
        self.upconv2 = layer_conv_transpose(in_channel=decoder_filters[2], 
                                            out_channel=decoder_filters[1], kernel_size=3, 
                                            stride=2, padding_mode=padding_mode)
        self.iconv2 = layer_conv(in_channel=decoder_filters[1]+decoder_filters[2], 
                                 out_channel=decoder_filters[1],  kernel_size=3, 
                                 stride=1, padding=0, padding_mode=padding_mode)
        
        self.upconv1 = layer_conv_transpose(in_channel=decoder_filters[1], 
                                            out_channel=decoder_filters[0], kernel_size=3, 
                                            stride=2, padding_mode=padding_mode)
        self.iconv1 = layer_conv(in_channel=decoder_filters[0], 
                                 out_channel=decoder_filters[0],  kernel_size=3, 
                                 stride=1, padding=0, padding_mode=padding_mode)
        
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(bias=True, in_channels=decoder_filters[0], out_channels=1, kernel_size=3, stride=1, padding=0),
            torch.nn.Softplus()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
    def _concat_and_pad(self, decoder_layer, encoder_layer, padding_mode):
        concat = torch.cat([decoder_layer, encoder_layer], axis=1)
        return torch.nn.functional.pad(concat, pad=(1,1,1,1), mode="constant")
        
    def forward(self, encoder_output):
        bottleneck, skip_connections = encoder_output
        (econv4, econv3, econv2, econv1) = skip_connections
        
        upconv5 = self.upconv5(bottleneck)
        cat_upconv5 = self._concat_and_pad(upconv5, econv4, self.padding_mode)
        iconv5 = self.iconv5(cat_upconv5)
    
        upconv4 = self.upconv4(iconv5)
        cat_upconv4 = self._concat_and_pad(upconv4, econv3, self.padding_mode)
        iconv4 = self.iconv4(cat_upconv4)
        
        upconv3 = self.upconv3(iconv4)
        cat_upconv3 = self._concat_and_pad(upconv3, econv2, self.padding_mode)
        iconv3 = self.iconv3(cat_upconv3)
        
        upconv2 = self.upconv2(iconv3)
        cat_upconv2 = self._concat_and_pad(upconv2, econv1, self.padding_mode)
        iconv2 = self.iconv2(cat_upconv2)

        upconv1 = self.upconv1(iconv2)
        upconv1 = torch.nn.functional.pad(upconv1, pad=[1,1,1,1], mode='constant')
        iconv1 = self.iconv1(upconv1)
        depth_input = torch.nn.functional.pad(iconv1, pad=[1,1,1,1], mode='constant')
        pred_depth = self.out(depth_input)
        return pred_depth
    

class DepthPredictor(torch.nn.Module):
    @property
    def _default_params(self):
        return {
            # Number of training steps over which the noise in randomized layer
            # normalization ramps up.
            'layer_norm_noise_rampup_steps': 10000,

            # Weight decay regularization of the network base.
            'weight_decay': 0.01,

            # If true, a learned scale factor will multiply the network's depth
            # prediction. This is useful when direct depth supervision exists.
            'learn_scale': False,

            # A boolean, if True, deconvolutions will be padded in 'REFLECT' mode,
            # otherwise in 'CONSTANT' mode (the former is not supported on TPU)
            'reflect_padding': False,
            'use_randomized_layer_norm': True
        }
    
    def __init__(self, params):
        super(DepthPredictor, self).__init__()
        self.params = self._default_params
        self.params.update(params or {})
        if self.params['use_randomized_layer_norm']:
            normalizer_fn = RandomizedLayerNorm
        else:
            normalizer_fn = _bn
        self.depth_encoder = ResNetEncoder(in_channel=3, normalizer_fn=normalizer_fn)
        self.depth_decoder = DepthDecoder(encoder_bottleneck_channel=512, reflect_padding=self.params['reflect_padding'])
        self.randomizedLayerNorm_modules = [module for module in self.depth_encoder.modules() if (type(module) is RandomizedLayerNorm)]


    def update_normalizer_fn(self, noise_stddev):
        """
        Function to change std of all RandomizedLayerNorm of the encoder during training.
        """
        for module in self.randomizedLayerNorm_modules:
            module.stddev = noise_stddev

    def forward(self, x, global_step=None):
        if self.training:
            noise_stddev = 0.5
            rampup_steps = self.params['layer_norm_noise_rampup_steps']
            if global_step is not None and rampup_steps > 0:
                # Add 1e-10 to avoid 0 std which causes error
                noise_stddev = noise_stddev * (min((global_step+1e-5) / rampup_steps, 1)**2)
            else: 
                # If not input global_step or rampup_steps = 0: then disable the random noise by setting std to be small
                noise_stddev = 1e-5
            self.update_normalizer_fn(noise_stddev)
        else:
            noise_stddev = 0
        encoder_output = self.depth_encoder(2*x-1)
        predicted_depth = self.depth_decoder(encoder_output)
        return [predicted_depth]