import torch
import torch.nn as nn
from torch.autograd import Variable
from unet import ConvBlock, EncoderBlock, DecoderBlock, BottleNeck, UNet
from collections import OrderedDict
import numpy as np
import gc

def simulate_convolution_trf(prev_trf, k, s, p):
    
    # get dimensions of the input image and the kernel
    m, n = len(prev_trf), len(prev_trf)
    prev_trf_padded = np.pad(prev_trf, ((p,p), (p,p), (0,0), (0,0)), 'edge')

    # compute output dimensions
    out_m = (m - k + 2*p) // s + 1
    out_n = (n - k + 2*p) // s + 1
    trf = np.empty((out_m, out_n, 2, 2), dtype=int)

    max_trf_size = 0

    # simulate convolution but join the receptive fields
    for i in range(out_m):
        for j in range(out_n):
            top_left = prev_trf_padded[i*s, j*s]
            bottom_right = prev_trf_padded[i*s+k-1, j*s+k-1]

            trf_size = bottom_right[1, 0] - top_left[0, 0] + 1
            max_trf_size = max(max_trf_size, trf_size)
            trf[i, j, 0] = top_left[0]
            trf[i, j, 1] = bottom_right[1]

    return trf, max_trf_size



def compute_trf(model, input_dim):
    n_modules = 0
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["skip"] = 1
    receptive_field["0"]["type"] = "input layer"
    receptive_field["0"]["trf"] = np.array([[[[j,i], [j,i]] for i in range(input_dim)]  for j in range(input_dim)], dtype=int)
    receptive_field["0"]["max_trf_size"] = 1

    skip_indices = []
    skip_trfs = dict()

    blocks = dict()

    encoding = True

    for module in model.modules():

        if type(module) not in [nn.Sequential, nn.ModuleList, nn.Module, nn.Linear, ConvBlock, EncoderBlock, DecoderBlock, BottleNeck, UNet]:
            n_modules += 1
        elif type(module) in [EncoderBlock, DecoderBlock, BottleNeck]:
            loc = n_modules + 1
            blocks[str(loc)] = module.__class__.__name__
            continue
        else:
            continue

        class_name = module.__class__.__name__
        module_idx = len(receptive_field)
        m_key = str(module_idx)
        prev_key = str(module_idx - 1)
        prev_trf = receptive_field[prev_key]["trf"]
        prev_max_rf = receptive_field[prev_key]["max_trf_size"]

        receptive_field[m_key] = OrderedDict()
        receptive_field[m_key]["type"] = class_name
        receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"]

        # COMPUTE RECEPTIVE FIELD OF CONVOLUTION WITH SKIP CONNECTION & UPSAMLE IN DECDOER BLOCK 
        if not encoding:
            print(len(blocks))
            assert class_name == "Conv2d"
            receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"] - 1
            encoding = True
            
            k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            s = module.stride if isinstance(module.stride, int) else module.stride[0]
            p = module.padding if isinstance(module.padding, int) else module.padding[0]
            p = (k - 1) // 2 if p == "s" else p

            # from skip
            skip_key = str(receptive_field[m_key]["skip"])
            skip_trf = skip_trfs[skip_key]

            # compute receptive field from upconv and skip concatenation
            concat_trf = np.zeros_like(prev_trf)
            for i in range(len(prev_trf)):
                for j in range(len(prev_trf[0])):
                    if prev_trf[i,j,0,0] <= skip_trf[i,j,0,0]:
                        concat_trf[i,j,0] = prev_trf[i,j,0]
                    else:
                        concat_trf[i,j,0] = skip_trf[i,j,0]
                    if prev_trf[i,j,1,1] >= skip_trf[i,j,1,1]:
                        concat_trf[i,j,1] = prev_trf[i,j,1]
                    else:
                        concat_trf[i,j,1] = skip_trf[i,j,1]
            
            # compute the trf of the convolution after the concatenation
            trf, max_trf_size = simulate_convolution_trf(prev_trf, k, s, p)
            receptive_field[m_key]["trf"] = trf
            receptive_field[m_key]["max_trf_size"] = max_trf_size

        else:
            prev_trf = receptive_field[prev_key]["trf"]

            # compute receptive field after maxpool
            if class_name == "MaxPool2d":
                receptive_field[m_key]["skip"] = receptive_field[prev_key]["skip"] + 1
                skip_trfs[str(receptive_field[prev_key]["skip"])] = receptive_field[prev_key]["trf"] # set to trf of previous ReLU
                skip_indices.append(prev_key)

                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                out_shape = prev_trf.shape[0] // k
                trf = np.zeros((out_shape, out_shape, 2, 2), dtype=int)
                max_trf_size = 0

                for i in range(out_shape):
                    for j in range(out_shape):
                        top_left = prev_trf[i*k, j*k]
                        bottom_right = prev_trf[i*k+k-1, j*k+k-1]
                        trf[i, j, 0] = top_left[0]
                        trf[i, j, 1] = bottom_right[1]
                        trf_size = bottom_right[1, 0] - top_left[0, 0] + 1
                        max_trf_size = max(max_trf_size, trf_size)

                receptive_field[m_key]["trf"] = trf
                receptive_field[m_key]["max_trf_size"] = max_trf_size

            # compute receptive field after convolution
            elif class_name == "Conv2d":
                k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                s = module.stride if isinstance(module.stride, int) else module.stride[0]
                p = module.padding if isinstance(module.padding, int) else module.padding[0]
                p = (k - 1) // 2 if p == "s" else p
                trf, max_trf_size = simulate_convolution_trf(prev_trf, k, s, p)
                receptive_field[m_key]["trf"] = trf
                receptive_field[m_key]["max_trf_size"] = max_trf_size

            # compute receptive field after deconvolution
            elif class_name == "ConvTranspose2d":
                encoding = False
                
                # compute receptive field of deconvolution
                kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                stride = module.stride if isinstance(module.stride, int) else module.stride[0]
                padding = module.padding if isinstance(module.padding, int) else module.padding[0]
                # output_padding = module.output_padding if isinstance(module.output_padding, int) else module.output_padding[0]

                
                # TODO
                receptive_field[m_key]["trf"] = prev_trf
                receptive_field[m_key]["max_trf_size"] = prev_max_rf

            # no changes in receptive field for batchnorm and relu
            elif class_name == "BatchNorm2d" or class_name == "ReLU":
                receptive_field[m_key]["trf"] = prev_trf
                receptive_field[m_key]["max_trf_size"] = prev_max_rf
            else:
                continue

    # print the receptive field for each layer
    line_new = "{:>5}  {:15}  {:>12} {:>10}".format("layer", "type", "max_trf_size", "skip conn.")
    print("-" * len(line_new))
    print(line_new)
    print("=" * len(line_new))
    
    for layer in receptive_field:
        skip_str = ""
        if int(layer) >= 1 and receptive_field[str(int(layer) - 1)]["type"] == "ConvTranspose2d":
            skip_str = str(receptive_field[layer]["skip"]) + " IN"
        elif receptive_field[layer]["type"] == "ReLU" and layer in skip_indices:
            skip_str = str(receptive_field[layer]["skip"]) + " OUT"

        line_new = "{:>5}  {:15}  {:>12} {:>10}".format(
            layer,
            receptive_field[layer]["type"],
            str(receptive_field[layer]["max_trf_size"]),
            skip_str
        )

        if layer in blocks:
            line_new = "\n" + blocks[layer] + "\n" + line_new

        print(line_new)

    print("-" * len(line_new))
    
    return receptive_field