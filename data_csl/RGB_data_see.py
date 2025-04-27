import pickle as pkl
import os
from tqdm import tqdm

RGB_dir = "/home/jlt/I3D_sign_features/domain_agnostic/csl_domain_agnostic/"
pose_dict_RGB = pkl.load(open("/home/jlt/I3D_sign_features/domain_agnostic/csl_domain_agnostic/S000054_P0008_T00.pkl", 'rb'))
pose_dict_Pose = pkl.load(open("/home/jlt/CSL/RTM_Keypoints/S000054_P0008_T00.pkl", 'rb'))

feature_num_list = []
RGB_dict_all = os.listdir(RGB_dir)
for RGB_file in tqdm(RGB_dict_all):
    RGB_file = os.path.join(RGB_dir, RGB_file)
    dict_RGB = pkl.load(open(RGB_file, 'rb'))
    feature_num, _ = dict_RGB['feature'].shape
    feature_num_list.append(feature_num)
feature_num_list = sorted(feature_num_list, reverse=True)
print(feature_num_list[:100])

# print(pose_dict_RGB.keys())
# print(type(pose_dict_RGB['name']))

# print(type(pose_dict_RGB['feature']))
# print(pose_dict_RGB['feature'].shape)

# print(pose_dict_Pose.keys())
# print(pose_dict_Pose['keypoints'].shape)
# print(type(pose_dict_Pose['keypoints']))
# print(type(pose_dict_Pose['img_list']))
# print(pose_dict_Pose['img_list'])