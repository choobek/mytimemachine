import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
import torchvision.transforms as transforms

from PIL import Image

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        # todo: https://github.com/google/mystyle/blob/main/utils/id_utils.py#L43
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    # def extract_feats(self, x):
    #     x = x[:, :, 35:223, 32:220]  # Crop interesting region
    #     x = self.face_pool(x)
    #     x_feats = self.facenet(x)
    #     return x_feats
    def extract_feats(self, img_path):
        img = self.transform_image(img_path)
        img = img.unsqueeze(0)
        img = img[:, :, 35:223, 32:220]
        img = self.face_pool(img)
        img_feats = self.facenet(img)
        return img_feats

    @staticmethod
    def transform_image(img_path):
        # if not torch.is_tensor(img):
        #     img = torch.from_numpy(img)
        # x = torch.unsqueeze(img, 0)
        # x = x / 127.5 - 1
        # x = torch.clamp(x, -1, 1)
        # if x.shape[-1] == 3:
        #     x = torch.permute(x, [0, 3, 1, 2])
        # return x # [b, c, h, w]
        img = Image.open(img_path).convert('RGB') # img is of shape (h, w, c)
        transform = transforms.Compose([
				transforms.Resize((256, 256)),
				# transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        img = transform(img)
        return img

# walk thorugh the images in the folder
import os
from collections import defaultdict
img_dir = '/playpen-nas-ssd/luchao/projects/SAM/age_dataset/pacino_30_80/aligned/train'
img_dict = defaultdict(list) # key: age, value: list of paths
age_dict = defaultdict(list) # key: age difference, value: list of similarities

# walk through the folder
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith('.jpeg'):
            img_path = os.path.join(root, file)
            age = int(file.split('.')[0].split(' ')[0])
            img_dict[age].append(img_path)


# calculate the similarity between each pair of images
# combinations between 2 ages
from itertools import combinations
comb = combinations(img_dict.keys(), 2)
print('total number of combinations:', len(list(comb)))
comb = combinations(img_dict.keys(), 2)
for age1, age2 in comb:
    for img1 in img_dict[age1]:
        for img2 in img_dict[age2]:
            feat_1 = IDLoss().extract_feats(img1)
            feat_2 = IDLoss().extract_feats(img2)
            sim = torch.nn.functional.cosine_similarity(feat_1, feat_2)
            age_diff = abs(age1 - age2)
            age_dict[age_diff].append(sim.item())

# also need to calculate the similarity between the same age
for age in img_dict.keys():
    for img1, img2 in combinations(img_dict[age], 2):
        feat_1 = IDLoss().extract_feats(img1)
        feat_2 = IDLoss().extract_feats(img2)
        sim = torch.nn.functional.cosine_similarity(feat_1, feat_2)
        age_dict[0].append(sim.item())

# save age_dict
import pickle
with open('age_dict.pkl', 'wb') as f:
    pickle.dump(age_dict, f)