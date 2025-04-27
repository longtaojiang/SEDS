import pickle as pkl

pose_dict = pkl.load(open("/data/jlt/CSL/hrnet_Keypoints/S000054_P0008_T00.pkl", 'rb'))
pose_dict_new = pkl.load(open("/data/jlt/CSL/RTM_Keypoints/S000054_P0008_T00.pkl", 'rb'))

print(pose_dict.keys())

print(pose_dict['keypoints'].shape)
print(type(pose_dict['keypoints']))

print(type(pose_dict['img_list']))
print(pose_dict['img_list'])

print(pose_dict_new['keypoints'].shape)
print(type(pose_dict_new['keypoints']))

print(type(pose_dict_new['img_list']))
print(pose_dict_new['img_list'])