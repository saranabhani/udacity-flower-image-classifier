import argparse
from utils import load_checkpoint, process_image, cat_2_name
import numpy as np
from PIL import Image
import os
import torch

parser = argparse.ArgumentParser(description='Predict flower name from an image.')
parser.add_argument('input', type=str, help='Path to the image')
parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to real names mapping file')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

if not os.path.exists(args.input):
    print("Image path doesn't exist: {}".format(args.input))
    raise FileNotFoundError

if not os.path.exists(args.checkpoint):
    print("checkpoint path doesn't exist: {}".format(args.checkpoint))
    raise FileNotFoundError
    
model = load_checkpoint('checkpoint.pth')
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)
img = process_image(args.input)
image_tensor = torch.from_numpy(img).type(torch.FloatTensor) if device.type == 'cpu' else torch.from_numpy(img).type(torch.cuda.FloatTensor)
model_input = image_tensor.unsqueeze(0)
model_input.to(device)
model.eval()

with torch.no_grad():
    output = model.forward(model_input)

    probs, indices = torch.exp(output).topk(args.top_k)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
if not os.path.exists(args.category_names):
    print("category_names path doesn't exist: {}".format(args.category_names))
    raise FileNotFoundError
cat2names = cat_2_name(args.category_names)
    
for cl,prob in zip(classes, probs):
    print(f"Class: {cat2names[cl]}, Probability: {prob}")

