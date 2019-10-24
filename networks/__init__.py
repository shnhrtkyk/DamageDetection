from .fcn import FCN8, FCN16, FCN32
from .erfnet import ERFNet
from .pspnet import PSPNet
from .segnet import SegNet
from .unet import UNet
from .network import AttU_Net, AttU_Net_Dual, AttU_Net_Dual_mod
from .utils import *

net_dic = {'erfnet' : ERFNet, 'fcn8' : FCN8, 'fcn16' : FCN16, 
                'fcn32' : FCN32, 'unet' : UNet, 'pspnet': PSPNet, 'segnet' : SegNet, 'att' :  AttU_Net, 'dmg' :  AttU_Net_Dual, 'dmg_mod' :  AttU_Net_Dual_mod}
                

def get_model(args):

    Net = net_dic[args.model]
    model = Net(args.num_classes)
    model.apply(weights_init)
    return model
