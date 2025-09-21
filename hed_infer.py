#!/usr/bin/env python

import getopt
import PIL
import PIL.Image
import sys
import os
import torch
import numpy as np
import PIL.Image

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'bsds500' # only 'bsds500' for now
args_strIn = './images/sample.png'
args_strOut = './out.png'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'in=',
    'out=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
    if strOption == '--in' and strArg != '': args_strIn = strArg # path to the input image
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + args_strModel + '.pytorch', file_name='hed-' + args_strModel).items() })
    # end

    def forward(self, ten_input):
        ten_input = ten_input * 255.0
        ten_input = ten_input - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=ten_input.dtype, device=ten_input.device).view(1, 3, 1, 1)

        ten_vgg_one = self.netVggOne(ten_input)
        ten_vgg_two = self.netVggTwo(ten_vgg_one)
        ten_vgg_thr = self.netVggThr(ten_vgg_two)
        ten_vgg_fou = self.netVggFou(ten_vgg_thr)
        ten_vgg_fiv = self.netVggFiv(ten_vgg_fou)

        ten_score_one = self.netScoreOne(ten_vgg_one)
        ten_score_two = self.netScoreTwo(ten_vgg_two)
        ten_score_thr = self.netScoreThr(ten_vgg_thr)
        ten_score_fou = self.netScoreFou(ten_vgg_fou)
        ten_score_fiv = self.netScoreFiv(ten_vgg_fiv)

        ten_score_one = torch.nn.functional.interpolate(input=ten_score_one, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        ten_score_two = torch.nn.functional.interpolate(input=ten_score_two, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        ten_score_thr = torch.nn.functional.interpolate(input=ten_score_thr, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        ten_score_fou = torch.nn.functional.interpolate(input=ten_score_fou, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        ten_score_fiv = torch.nn.functional.interpolate(input=ten_score_fiv, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ ten_score_one, ten_score_two, ten_score_thr, ten_score_fou, ten_score_fiv ], 1))
    # end
# end

netNetwork = None

##########################################################

def estimate(ten_input):
    global netNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if netNetwork is None:
        netNetwork = Network().to(device).train(False)

    int_width = ten_input.shape[2]
    int_height = ten_input.shape[1]

    #assert(int_width == 480)  # or comment this out if you want flexibility
    #assert(int_height == 320)

    return netNetwork(ten_input.to(device).view(1, 3, int_height, int_width))[0, :, :, :].cpu()


def load_image_safe(path):
    #Ensures RGB
    pil_img = PIL.Image.open(path)
    if pil_img.mode == "RGBA":
        background = PIL.Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img

def hed_from_path(input_path, output_path=None):
    pil_img = load_image_safe(input_path)

    # Convert PIL → numpy → tensor
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    np_img = np_img[:, :, ::-1].transpose(2, 0, 1)  # RGB→BGR, CHW
    ten_input = torch.FloatTensor(np.ascontiguousarray(np_img))

    ten_output = estimate(ten_input)

    ten_output = ten_output.squeeze().cpu().detach().numpy()  # (H, W)
    out_img = (np.clip(ten_output, 0.0, 1.0) * 255.0).astype(np.uint8)

    import cv2

    # 1. Quantize to discrete grayscale levels (instead of hard binary)
    levels = 32  # 4, 8, 16 or 32 depending on detail vs. stability
    step = 255 // levels
    out_img = (out_img // step) * step

    # 2. Optional: Dilate slightly to emphasize lines
    kernel = np.ones((1, 1), np.uint8)
    out_img = cv2.dilate(out_img, kernel, iterations=1)

    # Ensure 3 channels (RGB)
    if out_img.ndim == 2:
        out_img = np.stack([out_img] * 3, axis=-1)

    if not output_path:
        root, _ = os.path.splitext(input_path)
        # consistent PNG output
        output_path = f"{root}_hed.jpg"

    PIL.Image.fromarray(out_img, mode="RGB").save(output_path, format="JPEG")
    return output_path