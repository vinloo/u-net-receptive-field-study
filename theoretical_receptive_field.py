import torch
import torch.nn as nn
from torch.autograd import Variable
from unet import ConvBlock, EncoderBlock, DecoderBlock, BottleNeck
from collections import OrderedDict
import numpy as np


def compute_trf(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            prev_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()
            receptive_field[m_key]["type"] = class_name
            receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"]

            if not receptive_field["0"]["encoding"]:
                if class_name == "ConvTranspose2d":
                    receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"] - 1

                # from prev upconv 
                prev_j = receptive_field[prev_key]["j"]
                prev_r = receptive_field[prev_key]["r"]
                prev_start = receptive_field[prev_key]["start"]

                # from skip
                skip_key = str(receptive_field[m_key]["skip"])
                skip_j = receptive_field[skip_key]["j"]
                skip_r = receptive_field[skip_key]["r"]
                skip_start = receptive_field[skip_key]["start"]

                # TODO: COMPUTATION FOR RF OF UPCONVOLUTION
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                # COMPUTATION FOR RF OF CONVOLUTION
                prev_j = receptive_field[prev_key]["j"]
                prev_r = receptive_field[prev_key]["r"]
                prev_start = receptive_field[prev_key]["start"]

                if class_name == "MaxPool2d":
                    receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"] + 1

                if class_name == "Conv2d" or class_name == "MaxPool2d":
                    kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                    stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                    padding = module.padding if isinstance(module.padding, int) else module.padding[0]
                    dilation = module.dilation if isinstance(module.dilation, int) else module.dilation[0]

                    # Source of equations:
                    # https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
                    receptive_field[m_key]["j"] = prev_j * stride
                    receptive_field[m_key]["r"] = prev_r + ((kernel_size - 1) * dilation) * prev_j
                    receptive_field[m_key]["start"] = prev_start + ((kernel_size - 1) / 2 - padding) * prev_j
                elif class_name == "BatchNorm2d" or class_name == "ReLU":
                    receptive_field[m_key]["j"] = prev_j
                    receptive_field[m_key]["r"] = prev_r
                    receptive_field[m_key]["start"] = prev_start
                elif class_name == "ConvTranspose2d":
                    # only for the ConvTranspose2d in the FIRST decoder block
                    receptive_field["0"]["encoding"] = False
                    
                    # from skip
                    skip_key = str(receptive_field[m_key]["skip"])
                    skip_j = receptive_field[skip_key]["j"]
                    skip_r = receptive_field[skip_key]["r"]
                    skip_start = receptive_field[skip_key]["start"]

                    # TODO: COMPUTATION FOR RF OF FIRST UPCONVOLUTION
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError("module {} not ok".format(class_name))


            receptive_field[m_key]["input_shape"] = list(input[0].size())
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
            type(module) not in [nn.Sequential, nn.ModuleList, nn.Module, nn.Linear, ConvBlock, EncoderBlock, DecoderBlock, BottleNeck]
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))
        elif type(module) in [EncoderBlock, DecoderBlock, BottleNeck]:
            loc = str(len(hooks) - module.n_submodules + 1)
            blocks[str(loc)] = module.__class__.__name__

    dtype = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
    x = Variable(torch.rand(2, *input_size)).type(dtype)

    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["skip"] = 0
    receptive_field["0"]["encoding"] = True
    receptive_field["0"]["type"] = "input layer"
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []
    blocks = dict()

    model.apply(register_hook)
    model(x)

    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>5}  {:15}  {:>12}  {:>10} {:>10} {:>10} {:>7}".format("layer", "type", "map size", "start", "jump", "TRF", "skip")
    print(line_new)
    print("==============================================================================")
    
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4 or len(receptive_field[layer]["output_shape"]) == 5

        skip_str = ""
        if receptive_field[layer]["type"] == "ConvTranspose2d":
            skip_str = str(receptive_field[layer]["skip"]) + " IN"
        elif receptive_field[layer]["type"] == "MaxPool2d":
            skip_str = str(receptive_field[layer]["skip"]) + " OUT"

        line_new = "{:>5}  {:15}  {:>12}  {:>10} {:>10} {:>10} {:>7}".format(
            layer,
            receptive_field[layer]["type"],
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"])),
            skip_str
        )

        if layer in blocks:
            line_new = "\n" + blocks[layer] + "\n" + line_new

        # line_new += " " + str(receptive_field[layer]["skip"])
           
        print(line_new)

    print("==============================================================================")
    
    receptive_field["input_size"] = input_size
    return receptive_field


def get_pixel_trf(receptive_field_dict, layer: int, pixel_position):
    layer = str(layer)
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        feat_map_lim = rf_stats['output_shape'][2:]

        if np.any([pixel_position[idx] < 0 or pixel_position[idx] >= feat_map_lim[idx] for idx in range(len(pixel_position))]):
            raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
                
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in pixel_position]
        if len(input_shape) == 2:
            limit = input_shape
        else:
            limit = input_shape[1:3]
        rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]

        print("Receptive field size for layer %s, pixel_position %s,  is \n %s" % (layer, pixel_position, rf_range))
        return rf_range

    raise KeyError("Layer does not exist")
