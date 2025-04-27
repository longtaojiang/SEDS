import pickle as pkl

with open("data_csl/test.pkl", 'rb') as f:
    captions = pkl.load(f)

sentance_ids = captions.keys()
sentences_dict = {}
count = 0
for sentance_id in sentance_ids:
    # if len(captions[sentance_id]) > 1:
    #     print(sentance_id)
    #     print(captions[sentance_id])
    count = count + len(captions[sentance_id])
print(list(captions.keys())[:10])
print(captions['S000020'])
print(count)
print(len(captions))

pose_dict = pkl.load(open("/data/jlt/CSL/Keypoints_2d_mmpose/S000000_P0000_T00.pkl", 'rb'))

print('\n\n', pose_dict.keys())

print(len(pose_dict['keypoints']))
# print(pose_dict['keypoints'][0])
print(len(pose_dict['keypoints'][0]))

print(len(pose_dict['bbox']))
print(pose_dict['bbox'][0])

print(len(pose_dict['img_list']))
print(pose_dict['img_list'][:5])