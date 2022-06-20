import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
from models import build_model
import shutil, random, os
random.seed(0)
val_path = "/nobackup/yiwei/coco/images/val2017"
save_path = "/nobackup/yiwei/coco/images/vis/5_cross_att_head"
# save_path_2 = "/nobackup/yiwei/coco/images/20_conddetr_att"

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure()
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

checkpoint = torch.load("/nobackup/yiwei/CondDETR/output/5queries_cross_attention+LearnableRef_conddetr_r50_epoch50/checkpoint0049.pth")

model, criterion, postprocessors = build_model(checkpoint['args'])
model.load_state_dict(checkpoint['model'])
model.eval()

filenames = ['000000427338.jpg', '000000424975.jpg', '000000120853.jpg', '000000099810.jpg', '000000370818.jpg', '000000016502.jpg', '000000416256.jpg', '000000338905.jpg', '000000423617.jpg',  '000000295138.jpg', '000000066523.jpg', '000000031269.jpg', '000000014439.jpg', '000000453584.jpg', '000000009914.jpg', '000000210230.jpg', '000000306136.jpg', '000000263425.jpg', '000000288042.jpg', '000000396526.jpg', '000000016598.jpg', '000000323799.jpg', '000000159282.jpg', '000000474078.jpg', '000000564280.jpg', '000000175387.jpg', '000000223959.jpg', '000000492110.jpg', '000000186345.jpg', '000000106757.jpg', '000000495732.jpg', '000000495054.jpg', '000000218249.jpg', '000000537964.jpg', '000000050165.jpg', '000000163746.jpg', '000000020247.jpg', '000000500565.jpg', '000000287527.jpg', '000000365207.jpg', '000000068833.jpg', '000000499181.jpg', '000000521141.jpg', '000000434996.jpg', '000000281179.jpg', '000000214200.jpg']

# filenames = random.sample(os.listdir(val_path), 50)
for fname in filenames:
    srcpath = os.path.join(val_path, fname)
    im = Image.open(srcpath)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # plot_results(im, probas[keep], bboxes_scaled)
    # plt.savefig(os.path.join(save_path, fname))

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[5].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    if len(keep.nonzero()) == 0:
        continue

    # print(dec_attn_weights[5].shape)

    fig, axs = plt.subplots(ncols=6, nrows=9, squeeze=False, figsize=(22, 32))
    colors = COLORS * 100

    for row in range(9):
        for col in range(6):
            ax = axs[row][col]
            if col == 0:
                ax.imshow(im)
                # keep only predictions with 0.7+ confidence
                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.5

                # convert boxes from [0; 1] to image scales
                bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
                for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                            fill=False, color=c, linewidth=3))
                    cl = p.argmax()
                    text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                    ax.text(xmin, ymin, text, fontsize=15,
                            bbox=dict(facecolor='yellow', alpha=0.5))
                ax.set_title("Final outputs")
                ax.axis('off')
            else:
                if row <= 7:
                    ax.imshow(dec_attn_weights[0][0, row, col-1].view(h, w))
                    ax.set_title(f"Head {row+1}, Query {col}")
                else:
                    attens = dec_attn_weights[0].sum(dim=1) / 8
                    ax.imshow(attens[0][col-1].view(h, w))
                    ax.set_title(f"Averaged Query {col}")
                ax.axis('off')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fname))

    # for ax_i in axs.T:
    #     ax = ax_i[0]
    #     # print(dec_attn_weights[0].shape)
    #     if counter == 0:
    #         ax.imshow(im)
    #         for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
    #             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                     fill=False, color=c, linewidth=3))
    #             cl = p.argmax()
    #             text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
    #             ax.text(xmin, ymin, text, fontsize=15,
    #                     bbox=dict(facecolor='yellow', alpha=0.5))
    #         ax.axis('off')
    #     else:
    #         ax.imshow(dec_attn_weights[counter - 1][0, 0].view(h, w))
    #         ax.axis('off')
    #     counter += 1
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, fname))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                             fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(str(CLASSES[probas[idx].argmax()])+"   "+"{:.3f}".format(probas.max(-1).values[idx].item()))
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path_2, fname))
