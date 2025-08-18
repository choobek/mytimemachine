
import torch
from torch import nn
import torch.nn.functional as F

from configs.paths_config import model_paths
from models.dex_vgg import VGG
import os


# add system path
import sys
sys.path.append('/playpen-nas-ssd/luchao/projects/fpage')
sys.path.append('/playpen-nas-ssd/luchao/projects/face_parsing')
sys.path.append('/playpen-nas-ssd/luchao/projects/face_detection')
sys.path.append('/playpen-nas-ssd/luchao/projects/roi_tanh_warping')
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing.utils import label_colormap
from ibug.age_estimation import AgeEstimator
import numpy as np
import cv2
import torchvision
from PIL import Image


def tensor2im(var):
	'''
	var: tensor of shape (3, H, W)
	return: numpy array of shape (H, W, 3)
	'''
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

class AgingLoss(nn.Module):

    def __init__(self, opts):
        super(AgingLoss, self).__init__()
        self.age_net = VGG()
        ckpt = torch.load(model_paths['age_predictor'], map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.age_net.load_state_dict(ckpt)
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100
        self.opts = opts
        self.age_estimator = AgeEstimator()
        self.face_detector = RetinaFacePredictor(
            threshold=0.8,
            # threshold=0.5, # lower threshold to detect faces in low quality images
            device='cuda:0',
            model=(RetinaFacePredictor.get_model("mobilenet0.25")),
        )

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb, dim=1)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age
    
    def extract_ages(self, x):
        # x is a tensor of shape (b, c, h, w) [1, 3, 1024, 1024]
        # convert x for cv2
        # frame = x.permute(0, 2, 3, 1).cpu().numpy()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # faces = self.face_detector(frame, rgb=False)

        # frames = [torchvision.transforms.ToPILImage()(img) for img in x]
        # frames = [np.array(img) for img in frames]
        # frames = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in frames]

        frames = tensor2im(x[0])
        frames = np.array(frames)
        frames = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        # print('frames shape:', frames.shape)
        # print('mean:', np.mean(frames[0]))
        # print('std:', np.std(frames[0]))
        predicted_ages = []
        # for frame in frames:
        frame = frames
        faces = self.face_detector(frame, rgb=False)
        if len(faces) == 0:
            x = F.interpolate(x, size=(224, 224), mode='bilinear')
            predict_age_pb = self.age_net(x)['fc8']
            predicted_age = self.__get_predicted_age(predict_age_pb)
            # print('dex(no faces):') # use dex to predict age if no faces detected
            # print(predicted_age)
        else:
            age, masks = self.age_estimator.predict_img(frame, faces, rgb=False)
            predicted_age = torch.tensor(age[0].item())
            # print('fpage:')
            # print(predicted_age)
        predicted_ages.append(predicted_age)
        return torch.tensor(predicted_ages)
        return predicted_age
    
    def extract_ages_gt(self, x_path):
        '''
        Extracts the age from the ground truth image based on the file name
        '''
        def extract_age_gt(x_path):
            delimiter = ['_', '.', ' ']
            img_name = os.path.basename(x_path)
            for d in delimiter:
                img_name = img_name.replace(d, '/')
            age = int(img_name.split('/')[0])
            return age
        ages = []
        # determine single image path or batch of images
        if isinstance(x_path, str):
            age = extract_age_gt(x_path)
            ages.append(age)
        else:
            for path in x_path:
                age = extract_age_gt(path)
                ages.append(age)
        return torch.tensor(ages)
        # x = F.interpolate(x, size=(224, 224), mode='bilinear')
        # predict_age_pb = self.age_net(x)['fc8']
        # predicted_age = self.__get_predicted_age(predict_age_pb)
        # return predicted_age

    def forward(self, y_hat, y, target_ages, id_logs, label=None):
        n_samples = y.shape[0]

        if id_logs is None:
            id_logs = []

        input_ages = self.extract_ages(y) / 100.
        output_ages = self.extract_ages(y_hat) / 100.

        for i in range(n_samples):
            # if id logs for the same exists, update the dictionary
            if len(id_logs) > i:
                id_logs[i].update({f'input_age_{label}': float(input_ages[i]) * 100,
                                   f'output_age_{label}': float(output_ages[i]) * 100,
                                   f'target_age_{label}': float(target_ages[i]) * 100})
            # otherwise, create a new entry for the sample
            else:
                id_logs.append({f'input_age_{label}': float(input_ages[i]) * 100,
                                f'output_age_{label}': float(output_ages[i]) * 100,
                                f'target_age_{label}': float(target_ages[i]) * 100})

        loss = F.mse_loss(output_ages, target_ages)
        return loss, id_logs
