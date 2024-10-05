
import torch

class RefineMotionField(torch.nn.Module):
    
    def __init__(self, motion_field_channels, layer_channels, align_corners):
        super(RefineMotionField, self).__init__()
        self.align_corners = align_corners
#         self.motion_field_channels = motion_field_channels
        self.conv_output = torch.nn.Conv2d(bias=True, in_channels=motion_field_channels+layer_channels, out_channels=max(4, layer_channels), kernel_size=3, stride=1, padding=1)
        self.conv_input = torch.nn.Conv2d(bias=True, in_channels=motion_field_channels+layer_channels, out_channels=max(4, layer_channels), kernel_size=3, stride=1, padding=1)
        self.conv_output2 = torch.nn.Conv2d(bias=True, in_channels=max(4, layer_channels), out_channels=max(4, layer_channels), kernel_size=3, stride=1, padding=1)
        self.out = torch.nn.Conv2d(bias=False, in_channels=max(4, layer_channels)*2, out_channels=motion_field_channels, kernel_size=1, stride=1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, motion_field, layer):
        batch, channel, height, width = layer.shape 
        upsampled_motion_field = torch.nn.functional.interpolate(input=motion_field, 
                                                             size=[height, width], 
                                                             mode="bilinear",
                                                             align_corners=self.align_corners)
        conv_input = torch.cat([upsampled_motion_field, layer], dim=1)
        conv_output = torch.nn.functional.relu(self.conv_output(conv_input))
        conv_input = torch.nn.functional.relu(self.conv_input(conv_input))
        conv_output2 = torch.nn.functional.relu(self.conv_output2(conv_input))
        conv_output = torch.cat([conv_output, conv_output2], dim=1)
        out = self.out(conv_output)
        out = upsampled_motion_field + out
        return out



class MotionFieldNet(torch.nn.Module):
    
    def __init__(self, in_channels):
        """
            in_channels: image shape
            align_corners: align_corners in resize_bilinear. Only used in version 2.
            auto_mask: True to automatically masking out the residual translations
                by thresholding on their mean values.
        """

        super(MotionFieldNet, self).__init__()
        self.rot_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.trans_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.conv1 = torch.nn.Conv2d(bias=True, in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(bias=True, in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(bias=True, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(bias=True, in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(bias=True, in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(bias=True, in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(bias=True, in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0)
        
        # bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
        
        self.background_motion = torch.nn.Conv2d(bias=False, in_channels=1024, out_channels=6, kernel_size=1, stride=1, padding=0)
        
        self.residual_translation1 = torch.nn.Conv2d(bias=True, in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0)
        
        # Apply xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        
    def create_scales(self, constraint_minimum, rot_scale, trans_scale):
        """Creates variables representing rotation and translation scaling factors.
        Args:
          constraint_minimum: A scalar, the variables will be constrained to not fall
            below it.
        Returns:
          Two scalar variables, rotation and translation scale.
        """
        def constraint(x):
            return torch.nn.functional.relu(x-constraint_minimum) + constraint_minimum
        rot_scale = constraint(rot_scale)
        trans_scale = constraint(trans_scale)
        return rot_scale, trans_scale

    def tf_pad_stride2_1(self, x):
        return torch.nn.functional.pad(x, [0,1,0,1])

    def tf_pad_stride2_2(self, x):
        return torch.nn.functional.pad(x, [1,1,0,1])

    def forward(self, x):
        conv1 = torch.nn.functional.relu(self.conv1(self.tf_pad_stride2_1(x)))
        conv2 = torch.nn.functional.relu(self.conv2(self.tf_pad_stride2_1(conv1)))
        conv3 = torch.nn.functional.relu(self.conv3(self.tf_pad_stride2_1(conv2)))
        conv4 = torch.nn.functional.relu(self.conv4(self.tf_pad_stride2_1(conv3)))
        conv5 = torch.nn.functional.relu(self.conv5(self.tf_pad_stride2_1(conv4)))
        conv6 = torch.nn.functional.relu(self.conv6(self.tf_pad_stride2_2(conv5)))
        conv7 = torch.nn.functional.relu(self.conv7(self.tf_pad_stride2_2(conv6)))

        bottleneck = torch.mean(conv7, dim=[2,3], keepdim=True)

        background_motion = self.background_motion(bottleneck)
        rotation = background_motion[:, :3, 0, 0]
        background_translation = background_motion[:, 3:, :, :]

        rot_scale, trans_scale = self.create_scales(0.001, self.rot_scale, self.trans_scale)
        background_translation = background_translation * trans_scale
        rotation = rotation * rot_scale
            
        return rotation.reshape(-1,3), background_translation.reshape(-1,3)
        

# images = torch.randn(8, 3, 192, 640)
# torch_motionFieldNet = MotionFieldNet(in_channels=images.shape[1], align_corners=True, auto_mask=True)
# rotation, background_translation, residual_translation = torch_motionFieldNet(images)
