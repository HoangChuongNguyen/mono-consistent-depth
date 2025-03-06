
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
    
    def __init__(self, in_channels, align_corners=True, auto_mask=True):
        """
            in_channels: image shape
            align_corners: align_corners in resize_bilinear. Only used in version 2.
            auto_mask: True to automatically masking out the residual translations
                by thresholding on their mean values.
        """
        super(MotionFieldNet, self).__init__()
        self.rot_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.trans_scale = torch.nn.Parameter(torch.tensor(0.01))
        self.auto_mask = auto_mask
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
        
        self.residual_translation2 = RefineMotionField(motion_field_channels=3, layer_channels=1024, align_corners=align_corners)
        self.residual_translation3 = RefineMotionField(motion_field_channels=3, layer_channels=512, align_corners=align_corners)
        self.residual_translation4 = RefineMotionField(motion_field_channels=3, layer_channels=256, align_corners=align_corners)
        self.residual_translation5 = RefineMotionField(motion_field_channels=3, layer_channels=128, align_corners=align_corners)
        self.residual_translation6 = RefineMotionField(motion_field_channels=3, layer_channels=64, align_corners=align_corners)
        self.residual_translation7 = RefineMotionField(motion_field_channels=3, layer_channels=32, align_corners=align_corners)
        self.residual_translation8 = RefineMotionField(motion_field_channels=3, layer_channels=16, align_corners=align_corners)
        self.residual_translation  = RefineMotionField(motion_field_channels=3, layer_channels=in_channels, align_corners=align_corners)
        
        # ONLY ADD THESE TWO SO THAT THE NUMBER OF PARAMETERS MATCHES WITH THE TENSORFLOW VERSION
        # THESE PARAMTERS ARE NOT LEARNT AS WE DO NOT PREDICT INTRINSIC MATRIX. 
        self.focal_lengths, self.offsets = self.add_intrinsic_head()
        
        # Apply xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def add_intrinsic_head(self):
        focal_lengths = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=1)
        offsets = torch.nn.Conv2d(bias=False, in_channels=1024, out_channels=2, kernel_size=1)
        return focal_lengths, offsets
        
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
        
        residual_translation1 = self.residual_translation1(background_motion)
        residual_translation2 = self.residual_translation2(residual_translation1, conv7)
        residual_translation3 = self.residual_translation3(residual_translation2, conv6)
        residual_translation4 = self.residual_translation4(residual_translation3, conv5)
        residual_translation5 = self.residual_translation5(residual_translation4, conv4)
        residual_translation6 = self.residual_translation6(residual_translation5, conv3)
        residual_translation7 = self.residual_translation7(residual_translation6, conv2)
        residual_translation8 = self.residual_translation8(residual_translation7, conv1)
        residual_translation = self.residual_translation(residual_translation8, x)
        
        rot_scale, trans_scale = self.create_scales(0.001, self.rot_scale, self.trans_scale)
        background_translation = background_translation * trans_scale
        residual_translation = residual_translation * trans_scale
        rotation = rotation * rot_scale

        if self.auto_mask:
            sq_residual_translation = torch.norm(residual_translation, dim=1, keepdim=True)
            mean_sq_residual_translation = torch.mean(sq_residual_translation, dim=[0,2,3])
            mask_residual_translation = sq_residual_translation > mean_sq_residual_translation
            residual_translation = residual_translation * mask_residual_translation
            
        return rotation.reshape(-1,3), background_translation.reshape(-1,3)
        

# images = torch.randn(8, 3, 192, 640)
# torch_motionFieldNet = MotionFieldNet(in_channels=images.shape[1], align_corners=True, auto_mask=True)
# rotation, background_translation, residual_translation = torch_motionFieldNet(images)
