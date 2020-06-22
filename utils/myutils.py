import numpy as np
import cv2
import argparse
import imutils
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

model = models.resnet18(pretrained=True).cuda()
model.eval()
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
def make_mask(bbox, shape):
    x1, y1, x2, y2, x3, y3, x4, y4 = np.reshape(bbox, [-1])
    mask = np.ones(shape)
    a1 = -(y1 - y4) / (x1 - x4+0.0001)
    b1 = -(y1 - (y1 - y4) / (x1 - x4+0.0001) * x1)
    a2 = -(y1 - y2) / (x1 - x2+0.0001)
    b2 = -(y1 - (y1 - y2) / (x1 - x2+0.0001) * x1)
    a3 = -(y2 - y3) / (x2 - x3+0.0001)
    b3 = -(y2 - (y2 - y3) / (x2 - x3+0.0001) * x2)
    a4 = -(y3 - y4) / (x3 - x4+0.0001)
    b4 = -(y3 - (y3 - y4) / (x3 - x4+0.0001) * x3)
    for i in range(shape[1]):
        for j in range(shape[0]):
            if a1 > 0:
                if (j < -(a1 * i + b1)):
                    mask[j, i] = 0
            else:
                if (j > -(a1 * i + b1)):
                    mask[j, i] = 0
            if (j < -(a2 * i + b2)):
                mask[j, i] = 0
            if a3 > 0:
                if (j > -(a3 * i + b3)):
                    mask[j, i] = 0
            else:
                if (j < -(a3 * i + b3)):
                    mask[j, i] = 0
            if (j > -(a4 * i + b4)):
                mask[j, i] = 0
    return mask

def template_matching(img_gray, bbox, templates):
    template_matched = False
    sizes = np.zeros((len(templates), len(bbox)))
    for i in range(len(templates)):
        for j in range(len(bbox)):
            try:
                res = cv2.matchTemplate(img_gray[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]),
                                        np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])].astype(np.uint8),
                                        templates[i], cv2.TM_CCOEFF_NORMED)
            except:
                res = 0
            loc = np.where(res > 0.5)[0]
            sizes[i][j] = loc.size
    matched_bbox = np.argmax(np.sum(sizes, 0))
    if np.max(np.sum(sizes, 0)) > 100:
        template_matched = True
    return template_matched, bbox[matched_bbox]

# def template_matching_si(image, bbox, templates):
#     template_matched = False
#     sizes = np.zeros((len(templates), len(bbox)))
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     for i, template in enumerate(templates):
#         for j in range(len(bbox)):
#             # loop over the scales of the image
#             gray = img_gray[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]), np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])]
#             max_val = -5
#             for scale in np.linspace(1.0, 5.0, 10)[::-1]:
#                 resized = imutils.resize(gray, height=int(gray.shape[0] * scale))
#                 try:
#                     res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
#                 except:
#                     res = 0
#                 loc = np.where(res > 0.2)[0]
#                 loc_size = loc.size / template.size
#                 if loc_size > max_val:
#                     sizes[i][j] = loc_size
#                     max_val = loc_size
#
#     matched_bbox = np.argmax(np.sum(sizes, 0))
#     if np.max(np.sum(sizes, 0)) > 0:
#         template_matched = True
#     return template_matched, bbox[matched_bbox]


def template_matching_si(image, bbox, templates, main_template):
    template_matched = False
    sizes = np.zeros((len(templates), len(bbox)))
    for i, template in enumerate(templates):
        template = cv2.resize(template, (224, 224))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = Image.fromarray(template)
        t_img = Variable(normalize(to_tensor(scaler(template))).unsqueeze(0)).cuda()
        t_h = model(t_img)
        for j in range(len(bbox)):
            try:
                size = (int(bbox[j, 2, 1] - int(bbox[j, 0, 1]))) * (int(bbox[j, 2, 0]) - int(bbox[j, 0, 0]))
                if size > 10:
                    im_bbox = image[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]), np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])]
                    im_bbox = cv2.resize(im_bbox, (224, 224))
                    im_bbox = cv2.cvtColor(im_bbox, cv2.COLOR_BGR2RGB)
                    im_bbox = Image.fromarray(im_bbox)
                    im_bbox = Variable(normalize(to_tensor(scaler(im_bbox))).unsqueeze(0)).cuda()
                    image_h = model(im_bbox)
                    sizes[i][j] = cosine_similarity(np.reshape(image_h.cpu().detach().numpy(), [1, -1]), np.reshape(t_h.cpu().detach().numpy(), [1, -1]))[0][0]
                else:
                    sizes[i][j] = 0
            except:
                sizes[i][j] = 0
    matched_bbox = np.argmax(np.max(sizes, 0))
    if np.max(np.max(sizes, 0)) > 0.2:
        template_matched = True

    sizes = np.zeros(len(bbox))
    template = cv2.resize(main_template, (224, 224))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = Image.fromarray(template)
    t_img = Variable(normalize(to_tensor(scaler(template))).unsqueeze(0)).cuda()
    t_h = model(t_img)
    for j in range(len(bbox)):
        try:
            im_bbox = image[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]), np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])]
            im_bbox = cv2.resize(im_bbox, (224, 224))
            im_bbox = cv2.cvtColor(im_bbox, cv2.COLOR_BGR2RGB)
            im_bbox = Image.fromarray(im_bbox)
            im_bbox = Variable(normalize(to_tensor(scaler(im_bbox))).unsqueeze(0)).cuda()
            image_h = model(im_bbox)
            sizes[j] = cosine_similarity(np.reshape(image_h.cpu().detach().numpy(), [1, -1]), np.reshape(t_h.cpu().detach().numpy(), [1, -1]))[0][0]
        except:
            sizes[j] = 0
    return template_matched, bbox[matched_bbox], matched_bbox, np.max(sizes)


def template_matching_(image, bbox, templates):
    template_matched = 0
    sizes = np.zeros((len(templates), len(bbox)))
    for i, template in enumerate(templates):
        template = cv2.resize(template, (224, 224))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = Image.fromarray(template)
        t_img = Variable(normalize(to_tensor(scaler(template))).unsqueeze(0)).cuda()
        t_h = model(t_img)
        for j in range(len(bbox)):
            try:
                im_bbox = image[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]), np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])]
                im_bbox = cv2.resize(im_bbox, (224, 224))
                im_bbox = cv2.cvtColor(im_bbox, cv2.COLOR_BGR2RGB)
                im_bbox = Image.fromarray(im_bbox)
                im_bbox = Variable(normalize(to_tensor(scaler(im_bbox))).unsqueeze(0)).cuda()
                image_h = model(im_bbox)
                sizes[i][j] = cosine_similarity(np.reshape(image_h.cpu().detach().numpy(), [1, -1]), np.reshape(t_h.cpu().detach().numpy(), [1, -1]))[0][0]
            except:
                sizes[i][j] = 0
    if np.max(np.max(sizes, 0)) > 0.5:
        template_matched = 1
    return template_matched


def template_matching_increase(image, bbox, main_template, old_score):
    template_matched = 0
    sizes = np.zeros(len(bbox))
    template = cv2.resize(main_template, (224, 224))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template = Image.fromarray(template)
    t_img = Variable(normalize(to_tensor(scaler(template))).unsqueeze(0)).cuda()
    t_h = model(t_img)
    for j in range(len(bbox)):
        try:
            im_bbox = image[np.maximum(int(bbox[j, 0, 1]), 0):int(bbox[j, 2, 1]), np.maximum(int(bbox[j, 0, 0]), 0):int(bbox[j, 2, 0])]
            im_bbox = cv2.resize(im_bbox, (224, 224))
            im_bbox = cv2.cvtColor(im_bbox, cv2.COLOR_BGR2RGB)
            im_bbox = Image.fromarray(im_bbox)
            im_bbox = Variable(normalize(to_tensor(scaler(im_bbox))).unsqueeze(0)).cuda()
            image_h = model(im_bbox)
            sizes[j] = cosine_similarity(np.reshape(image_h.cpu().detach().numpy(), [1, -1]), np.reshape(t_h.cpu().detach().numpy(), [1, -1]))[0][0]
        except:
            sizes[j] = 0
    cur_score = np.max(sizes)
    # print("old_score", old_score)
    # print("cur_score", cur_score)
    if cur_score > old_score:
        template_matched = 1
    return template_matched
