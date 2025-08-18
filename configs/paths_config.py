dataset_paths = {
    'celeba_test': '/playpen-nas-ssd/luchao/data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img',
    'ffhq': '/playpen-nas-ssd/luchao/data/ffhq-dataset/images1024x1024',
}

model_paths = {
    'pretrained_psp': 'pretrained_models/psp_ffhq_encode.pt',
    # 'pretrained_psp': '/playpen-nas-ssd/luchao/projects/e4e/pretrained_models/e4e_ffhq_encode.pt',
    'ir_se50': '/playpen-nas-ssd/luchao/projects/SAM/pretrained_models/model_ir_se50.pth',
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': '/playpen-nas-ssd/luchao/projects/SAM/pretrained_models/dex_age_classifier.pth'
}
