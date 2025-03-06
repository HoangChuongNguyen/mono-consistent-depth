
from .corl2020.depth_nets import DepthPredictor as CorlDepthNet
# from .corl2020.pose_net import MotionFieldNet as CorlPoseNet
from .corl2020.pose_net_old import MotionFieldNet as CorlPoseNet

# from .packnet.depth.PackNet01 import PackNet01 
from .packnet.depth.PackNet01 import PackNet01
from .packnet.pose.PoseResNet import PoseResNet as PackNetPoseResNet

from .manydepth.depth_net import ManyDepthNet, MonoDepthNet
from .manydepth.pose_net import ManyPoseNet
from .manydepth.resnet_encoder import ResnetEncoderMatching as ManyDepthEncoder
from .manydepth.resnet_encoder import ResnetEncoder as ManyDepthResnetEncoder
from .manydepth.depth_decoder import DepthDecoder as ManyDepthDecoder
from .manydepth.pose_decoder import PoseDecoder as ManyPoseDecoder


from .diffnet.depthnet import DiffDepthNet
from .diffnet import test_hr_encoder as DiffNetDepthEncoder
from .diffnet.HR_Depth_Decoder import HRDepthDecoder as DiffNetHRDepthDecoder
from .diffnet.resnet_encoder import ResnetEncoder as DiffNetResnetEncoder
from .diffnet.pose_decoder import PoseDecoder as DiffNetPoseDecoder
from .diffnet.pose_net import DiffPoseNet

from .brnet.resnet_encoder import ResnetEncoder as BrNetResnetEncoder
from .brnet.depth_decoder import DepthDecoder as BrNetDepthDecoder
from .brnet.pose_decoder import PoseDecoder as BrNetPoseDecoder
from .brnet.pose_net import BrPoseNet
from .brnet.depth_net import BrDepthNet
