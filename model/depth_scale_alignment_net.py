
import torch
import torch.nn as nn
from torchvision.models import resnet18

class DepthScaleAlignmentNet(nn.Module):
    def __init__(self, replace_bn_by_gn):
        super(DepthScaleAlignmentNet, self).__init__()

        # Load the pre-trained ResNet model for both branches
        self.all_feature_extractor = resnet18(pretrained=True)

        # Adjust the first layer to accept single-channel input
        self.all_feature_extractor.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the final fully connected layer to get feature maps
        # self.dynamic_feature_extractor = nn.Sequential(*list(self.dynamic_feature_extractor.children())[:-1])
        # self.static_feature_extractor = nn.Sequential(*list(self.static_feature_extractor.children())[:-1])
        self.all_feature_extractor = nn.Sequential(*list(self.all_feature_extractor.children())[:-1])

        if replace_bn_by_gn:
            self.all_feature_extractor = self.convert_bn_to_gn(self.all_feature_extractor)

        # Flatten and Dense Layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def convert_bn_to_gn(self, module):
        module_output = module
        if isinstance(module, nn.BatchNorm2d):
            module_output = nn.GroupNorm(1, module.num_features, eps=module.eps, affine=module.affine)
            if module.affine:
                module_output.weight.data = module.weight.data.clone().detach()
                module_output.bias.data = module.bias.data.clone().detach()
        for name, child in module.named_children():
            module_output.add_module(name, self.convert_bn_to_gn(child))
        return module_output

    def forward(self, images, object_masks,static_depths,  object_depths):
        combined_depth = (1-object_masks)*static_depths + object_masks*object_depths
        rgbd_image = torch.concat([images, combined_depth, object_depths], dim=1)
        # # Pass through the first feature extractor
        # dynamic_features = self.dynamic_feature_extractor(object_depths)
        # # Pass through the second feature extractor
        # static_features = self.static_feature_extractor(static_depths)
        # Pass through the third feature extractor
        all_features = self.all_feature_extractor(rgbd_image)
        # Concatenate along the depth dimension
        features = all_features
        features = features.view(features.shape[0], features.shape[1])
        # Flatten and pass through dense layers
        scales = self.fc(features)
        scales = scales  # Force the predicted scale to be >= 0
        return scales