from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import random
import math

LEN_KPS = 10
THRE = 0.5

class ph_DataLoader_pose(Dataset):
    """PH dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            features_RGB_path,
            tokenizer,
            max_words=30,
            feature_len=64,
            max_length_frames=300,
            slide_windows=16,
            windows_stride=1,
            args=None
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.features_RGB_path = features_RGB_path
        self.interval = args.interval
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.feature_len = feature_len
        self.subset = subset
        self.slide_windows = slide_windows
        self.windows_stride = windows_stride
        self.max_length_frames = max_length_frames
        self.threshold = args.threshold
        self.frames_threshold = args.frames_threshold
        self.crop_img_size = np.array([[args.crop_size, args.crop_size]], dtype=np.float32)

        assert self.subset in ["train", "dev", "test"]
        sentance_id_path_dict = {}
        sentance_id_path_dict["train"] = os.path.join(self.data_path, "train.pkl")
        sentance_id_path_dict["test"] = os.path.join(self.data_path, "test.pkl")
        with open(sentance_id_path_dict[self.subset], 'rb') as f:
            captions = pkl.load(f)

        self.captions=captions

        sentance_ids = captions.keys()
        sentences_dict = {}
        for sentance_id in sentance_ids:
            text=captions[sentance_id]['text']
            sentences_dict[sentance_id] = text
        self.sentences_dict = sentences_dict

        self.sample_len = 0
        self.video_dict = {}
        self.video_RGB_dict = {}
        self.cut_off_points = []

        for sentance_id in sentance_ids:
            video=captions[sentance_id]
            video_name=video['video_name']
            self.video_dict[len(self.video_dict)] = (sentance_id, os.path.join(self.features_path, video_name)+'.pkl')
            self.video_RGB_dict[len(self.video_RGB_dict)] = (sentance_id, os.path.join(os.path.join(self.features_RGB_path, subset), video_name)+'.pkl')

            self.cut_off_points.append(len(self.video_dict))

        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(self.video_dict)
            assert len(self.cut_off_points) == self.sentence_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Sentance number: {}".format(len(self.sentences_dict)))
        print("Total Paire: {}".format(len(self.video_dict)))

        self.sample_len = len(self.video_dict)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):

        video_feature, video_mask = self._get_rawvideo(idx)
        sample, sentance_id = self._get_pose(idx)

        sample_text = self._get_text(sentance_id)
        sample['text'] = sample_text
        sample['RGB'] = video_feature
        assert torch.sum(video_mask) == torch.sum(sample['right']['pose_mask']), print(torch.sum(video_mask), print(torch.sum(sample['right']['pose_mask'])), idx)

        return sample
    
    def _get_rawvideo(self, vedio_index):
        feature_len = self.feature_len
        item = self.video_RGB_dict[vedio_index]
        _, video_file_path = item

        video_feature = torch.zeros((1024, feature_len, 1))
        video_mask = torch.ones(feature_len + 1, dtype=torch.long)
        video_mask[0] = 0

        with open(video_file_path, 'rb') as f:
            item = pkl.load(f)
            video_feature_pre = item['feature']

        video_feature_pre=torch.Tensor(video_feature_pre).transpose(0, 1)
        video_feature_pre=video_feature_pre.view(video_feature_pre.shape[0], -1, 1)

        video_len=video_feature_pre.shape[1]


        if video_len>=feature_len:
            choosen_idx=range(video_len)
            choosen_idx=list(choosen_idx)
            ValueError("RGB data has problem!!!")
        else:
            choosen_idx=range(video_len)
            choosen_idx=list(choosen_idx)
        for i in range(len(choosen_idx)):
            video_feature[:,i,:]=video_feature_pre[:,choosen_idx[i],:]
            video_mask[i+1]=0

        return video_feature, video_mask

    def _get_text(self, sentence_ids):
        k = 1
        choice_sentance_ids=[sentence_ids]
        pairs_text = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.longlong)

        pairs_segment = np.zeros((k, self.max_words), dtype=np.longlong)
        
        for i, sentance_id in enumerate(choice_sentance_ids):
            
            text = self.sentences_dict[sentance_id]

            words = self.tokenizer.tokenize(text)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            words_index=[0]
            if len(words) > total_length_with_CLS:
                #selected_index = list(np.arange(len(words)-1))
                all_index = list(np.linspace(1, len(words) - 1, total_length_with_CLS-1, dtype=int))
                # all_index = list(np.random.randint(1, len(words) - 1, total_length_with_CLS - 1, dtype=int))
                selected_index=sorted(all_index)
                words_index+=selected_index
                words=list(np.array(words)[words_index])
                # words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            sample = {}
            sample["pairs_text"] = torch.from_numpy(pairs_text)
            sample["pairs_mask"] = torch.from_numpy(pairs_mask)
            sample["pairs_segment"] = torch.from_numpy(pairs_segment)

        return sample

    def _get_pose_clips(self, pose_sample):

        frames_num, joints_num, coor_num = pose_sample.shape
        slide_windows = self.slide_windows
        windows_stride = self.windows_stride
        feature_len = self.feature_len
        
        num_clips = math.ceil((frames_num - slide_windows) / windows_stride) + 1#clips in video

        if num_clips<=0:
            num_clips=1

        if frames_num < slide_windows:
            for i in range(slide_windows-frames_num):
                last_frames = pose_sample[-1, :, :].unsqueeze(0)
                pose_sample = torch.cat([pose_sample, last_frames], dim=0)

        t_beg = []
        pose_mask = torch.ones(feature_len + 1, dtype=torch.long)
        pose_mask[0] = 0
 
        for j in range(num_clips):
            actual_clip_length = min(self.slide_windows, frames_num - j * windows_stride)
            if actual_clip_length == self.slide_windows:
                t_beg.append(0 + j * windows_stride)
            else:

                if frames_num - self.slide_windows>=0:
                    t_beg.append(frames_num - self.slide_windows)
                else:
                    t_beg.append(0)
            
        if num_clips > feature_len:
            all_index=list(np.linspace(0, num_clips-1, feature_len, dtype=int))
            choosen_idx = sorted(all_index)
            num_clips = feature_len
        else:
            choosen_idx=range(num_clips)
            choosen_idx=list(choosen_idx)

        clips_start = torch.ones(feature_len, dtype=torch.long) * -1

        for i in range(len(choosen_idx)):
            clip_idx = choosen_idx[i]
            clip_start = t_beg[clip_idx]
            clips_start[i] = clip_start

            pose_mask[i+1]=0

        return_dict = {}
        return_dict['pose_sample'] = pose_sample
        return_dict['pose_mask'] = pose_mask
        return_dict['clips_start'] = clips_start

        return return_dict

    def _get_pose(self, ids):
        sentance_id, video_file_path = self.video_dict[ids]

        video_data = pkl.load(open(video_file_path, 'rb'))
        ori_img_size = np.array([210, 260], dtype=np.float32)
        total_frame_list = self.GetTotalFrameList(video_data, ori_img_size)

        sample = {}
        sample['right'] = self._get_single_hand('right' ,video_data, ori_img_size, total_frame_list.copy())
        sample['left'] = self._get_single_hand('left', video_data, ori_img_size, total_frame_list.copy())
        sample['body'] = self._get_body_pose(video_data, ori_img_size, total_frame_list)

        assert sample['right']['kp2d'].shape[0] == sample['left']['kp2d'].shape[0] == sample['body']['body_pose'].shape[0]

        sample['right'] = self._get_pose_clips(sample['right']['kp2d'])
        sample['left'] = self._get_pose_clips(sample['left']['kp2d'])
        sample['body'] = self._get_pose_clips(sample['body']['body_pose'])

        assert torch.sum(sample['right']['pose_mask']) == torch.sum(sample['left']['pose_mask']) == torch.sum(sample['body']['pose_mask'])
        assert torch.sum(sample['right']['clips_start']) == torch.sum(sample['left']['clips_start']) == torch.sum(sample['body']['clips_start'])

        return sample, sentance_id
    
    def _get_body_pose(self, video_data, ori_img_size, total_frame_list):
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']
        assert video_joints.shape[0] == len(frame_list)

        if len(frame_list) > self.max_length_frames:
            interval = max(self.interval, math.ceil(len(frame_list) / self.max_length_frames))
            frame_index = slice(0, len(video_joints), interval)
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]

        root_pos_gt = []
        root_pos_valid = []
        for frame in total_frame_list:
            i = frame_list.index(frame)
            skeleton = video_joints[i][:, 0:2]
            confidences = video_joints[i][:, 2]
            body_pose, body_conf = self.get_kp2ds(skeleton, confidences, 0.0, part='body')

            gt = body_pose / ori_img_size * self.crop_img_size

            root_pos_gt.append(gt)
            root_pos_valid.append(body_conf)

        root_pos_gt = np.stack(root_pos_gt, axis=0).astype(np.float32)
        root_pos_valid = np.stack(root_pos_valid, axis=0).astype(np.float32)
        root_pos_gt = torch.from_numpy(root_pos_gt).float()
        root_pos_valid = torch.from_numpy(root_pos_valid).float()

        sample = {
            'body_pose': root_pos_gt,
            # 'body_pose_conf': root_pos_valid,
        }

        return sample
    
    def _get_single_hand(self, hand_side, video_data, ori_img_size, total_frame_list):
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']
        assert video_joints.shape[0] == len(frame_list)

        if len(frame_list) > self.max_length_frames:
            interval = max(self.interval, math.ceil(len(frame_list) / self.max_length_frames))
            frame_index = slice(0, len(video_joints), interval)
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]
        video_joints_update, bbxes, frame_list_update = self.crop_hand(hand_side, video_joints, ori_img_size, frame_list)

        if len(total_frame_list) == 0 or frame_list_update is None:
            raise ValueError('len(total_frame_list) == 0 or frame_list_update is None')

        kp2ds_total = []
        conf = []
        for i in range(len(video_joints_update)):
            frame_id = total_frame_list.index(frame_list_update[i])
            if frame_id != i:
                # raise ValueError('frame_id != i')
                del total_frame_list[i:frame_id]
                length = frame_id - i
                kp2d = torch.zeros((length, 21, 2), dtype=torch.float32)
                kp2ds_total.append(kp2d)
                conf.append(torch.zeros_like(kp2d))

            skeleton = video_joints_update[i][:, 0:2]
            confidences = video_joints_update[i][:, 2]
            kp2ds, confidence = self.get_kp2ds(skeleton, confidences, self.threshold, part=hand_side)
            trans = np.array([[bbxes[i][0], bbxes[i][2]]], dtype=np.float32)
            scale = np.array(
                [[bbxes[i][1] - bbxes[i][0], bbxes[i][3] - bbxes[i][2]]],
                dtype=np.float32)
            assert scale[0, 1] > 0.0 and scale[0, 0] > 0.0
            kp2ds = (kp2ds - trans) / scale * self.crop_img_size
            kp2ds = np.where(kp2ds > 0.0, kp2ds, 0.0)
            if hand_side == 'left':
                kp2ds[:, 0] = self.crop_img_size[0, 0] - kp2ds[:, 0]
                kp2ds = np.where(kp2ds < 255.0, kp2ds, 0.0)

            kp2ds_total.append(kp2ds[np.newaxis, :, :])
            conf.append(confidence[np.newaxis, :, :])

        # existing condition that only last several frames don't exist
        if len(total_frame_list) != len(frame_list_update):
            # raise ValueError('len(total_frame_list) != len(frame_list_update)')
            length = len(total_frame_list) - len(frame_list_update)
            kp2d = torch.zeros((length, 21, 2), dtype=torch.float32)
            kp2ds_total.append(kp2d)
            conf.append(torch.zeros_like(kp2d))

        kp2ds_total = np.concatenate(kp2ds_total, axis=0).astype(np.float32)
        conf = np.concatenate(conf, axis=0).astype(np.float32)
        kp2ds_total = torch.from_numpy(kp2ds_total).float()
        conf = torch.from_numpy(conf).float()
        sample = {
            'kp2d': kp2ds_total,
            # 'flag_2d': conf,
        }
        return sample

    def get_kp2ds(self, skeleton, conf, threshold, part):
        if part == 'left':
            hand_kp2d = skeleton[91:112, :]
            confidence = conf[91:112]
        elif part == 'right':
            hand_kp2d = skeleton[112:133, :]
            confidence = conf[112:133]
        elif part == 'body':
            hand_kp2d = skeleton[[0, 5, 7, 9, 6, 8, 10], :]
            confidence = conf[[0, 5, 7, 9, 6, 8, 10]]
        else:
            raise Exception('wrong hand_side type')
        confidence = np.where(confidence > threshold, confidence, 0.0)
        indexes = np.where(confidence < threshold)[0].tolist()
        for i in range(len(indexes)):
            hand_kp2d[indexes[i]] = np.zeros((1, 2), dtype=np.float32)
        confidence = np.tile(confidence[:, np.newaxis], (1, 2))
        return hand_kp2d, confidence

    def GetTotalFrameList(self, video_data, ori_img_size):
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']
        assert video_joints.shape[0] == len(frame_list)

        if len(frame_list) > self.max_length_frames:
            interval = max(self.interval, math.ceil(len(frame_list) / self.max_length_frames))
            start = random.randint(0, interval)
            frame_index = slice(0, len(video_joints), interval)
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]
            
        _, _,  frame_list_update_right = self.crop_hand('right', video_joints, ori_img_size, frame_list)
        _, _, frame_list_update_left = self.crop_hand('left', video_joints, ori_img_size, frame_list)
        if frame_list_update_left is None:
            total_frame_list = frame_list_update_right
        elif frame_list_update_right is None:
            total_frame_list = frame_list_update_left
        else:
            total_frame_list = list(set(frame_list_update_right + frame_list_update_left))
        total_frame_list = sorted(total_frame_list, key=lambda x: int(x.split('.')[0][6:]))
        return total_frame_list

    def crop_hand(self, hand_side, video_joints, ori_img_size, frames_name):
        video_joints_new = []
        frames_name_new = []
        bbxes = []
        for i in range(len(frames_name)):
            skeleton = video_joints[i, :, 0:2]
            confidence = video_joints[i, :, 2]
            usz, vsz = [ori_img_size[0], ori_img_size[1]]
            minsz = min(usz, vsz)
            if hand_side == 'right':
                right_keypoints = skeleton[112:133, :]
                kp_visible = (confidence[112:133] > self.frames_threshold)
                uvis = right_keypoints[kp_visible, 0]
                vvis = right_keypoints[kp_visible, 1]
            elif hand_side == 'left':
                left_keypoints = skeleton[91:112, :]
                kp_visible = (confidence[91:112] > self.frames_threshold)
                uvis = left_keypoints[kp_visible, 0]
                vvis = left_keypoints[kp_visible, 1]
            else:
                raise ValueError('wrong hand side')
            if len(uvis) < LEN_KPS:
                bbx = self.elbow_hand(hand_side, skeleton, confidence, ori_img_size)
                if bbx is None:
                    continue
                else:
                    bbxes.append(bbx)
                    video_joints_new.append(video_joints[i])
                    frames_name_new.append(frames_name[i])
            else:
                umin = min(uvis)
                vmin = min(vvis)
                umax = max(uvis)
                vmax = max(vvis)

                B = round(2.2 * max([umax - umin, vmax - vmin]))

                us = 0
                ue = usz - 1
                vs = 0
                ve = vsz - 1
                umid = umin + (umax - umin) / 2
                vmid = vmin + (vmax - vmin) / 2

                if (B < minsz - 1):
                    us = round(max(0, umid - B / 2))
                    ue = us + B
                    if (ue > usz - 1):
                        d = ue - (usz - 1)
                        ue = ue - d
                        us = us - d
                    vs = round(max(0, vmid - B / 2))
                    ve = vs + B
                    if (ve > vsz - 1):
                        d = ve - (vsz - 1)
                        ve = ve - d
                        vs = vs - d
                if (B >= minsz - 1):
                    B = minsz - 1
                    if usz == minsz:
                        vs = round(max(0, vmid - B / 2))
                        ve = vs + B
                        if (ve > vsz - 1):
                            d = ve - (vsz - 1)
                            ve = ve - d
                            vs = vs - d
                    if vsz == minsz:
                        us = round(max(0, umid - B / 2))
                        ue = us + B

                        if (ue > usz - 1):
                            d = ue - (usz - 1)
                            ue = ue - d
                            us = us - d
                us = int(us)
                vs = int(vs)
                ue = int(ue)
                ve = int(ve)
                bbx = [us, ue, vs, ve]
                bbxes.append(bbx)
                video_joints_new.append(video_joints[i])
                frames_name_new.append(frames_name[i])

        bbxes = np.array(bbxes, dtype=np.float32)
        if len(bbxes) == 0:
            return None, None, None
        return video_joints_new, bbxes, frames_name_new

    def elbow_hand(self, hand_side, pose_keypoints, confidence, ori_img_size):
        right_hand = pose_keypoints[[6, 8, 10]]
        left_hand = pose_keypoints[[5, 7, 9]]
        ratioWristElbow = 0.33
        detect_result = []
        usz, vsz = [ori_img_size[0], ori_img_size[1]]
        if hand_side == 'right':
            has_right = np.sum(confidence[[6, 8, 10]] < THRE) == 0
            if not has_right:
                return None
            x1, y1 = right_hand[0][:2]
            x2, y2 = right_hand[1][:2]
            x3, y3 = right_hand[2][:2]

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2  # width = height

            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > usz: width1 = usz - x
            if y + width > vsz: width2 = vsz - y
            width = min(width1, width2)
            detect_result.append([int(x), int(y), int(width)])

        elif hand_side == 'left':
            has_left = np.sum(confidence[[5, 7, 9]] < THRE) == 0
            if not has_left:
                return None
            x1, y1 = left_hand[0][:2]
            x2, y2 = left_hand[1][:2]
            x3, y3 = left_hand[2][:2]

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2  # width = height
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > usz: width1 = usz - x
            if y + width > vsz: width2 = vsz - y
            width = min(width1, width2)
            detect_result.append([int(x), int(y), int(width)])

        x, y, width = int(x), int(y), int(width)
        return [x, x + width, y, y + width]


def my_collate_fn_body_pose(batch):
    '''
    processing body pose data
    :param batch: list type, input body pose data with batch size
    :param padding: None
    :return: batch: the padding sequence for different videos
    '''
    len_video = [x['pose_sample'].shape[0] for x in batch]
    max_len = max(len_video)
    for i in range(len(len_video)):
        if max_len - len_video[i] > 0:
            sample = batch[i]
            pose_sample = sample['pose_sample']
            pose_sample_new = torch.cat((pose_sample, torch.zeros_like(pose_sample[-1]).expand(max_len - len_video[i], -1, -1)), dim=0).float()
            batch[i]['pose_sample'] = pose_sample_new
    return batch

def my_collate_fn_single_hand(batch, padding=6):
    '''
    processing single hand data
    :param batch: list type, input single hand data with batch size
    :param padding: None
    :return: batch: the padding sequence for different videos; max_Len: the maximum length for input data on single hand
    '''
    len_video = [x['pose_sample'].shape[0] for x in batch]
    max_len = max(len_video)
    for i in range(len(len_video)):
        if max_len - len_video[i] > 0:
            sample = batch[i]
            pose_sample = sample['pose_sample']
            pose_sample_new = torch.cat((pose_sample, torch.zeros_like(pose_sample[-1]).expand(max_len - len_video[i], -1, -1)), dim=0).float()
            batch[i]['pose_sample'] = pose_sample_new

    return batch, max_len


def ph_pose_collate_fn(batch, padding=6):
    '''
    process input data using specific methods, padding time sequence for different videos
    :param batch: list type, input data with batch size
    :param padding: None
    :return: dict type, input data for SignBert model
    '''
    batch_RGB = []
    batch_right = []
    batch_left = []
    batch_body = []
    batch_pairs_text = []
    batch_pairs_mask = []
    batch_pairs_segment = []

    for i in range(len(batch)):
        batch_RGB.append(batch[i]['RGB'])
        batch_right.append(batch[i]['right'])
        batch_left.append(batch[i]['left'])
        batch_body.append((batch[i]['body']))

        batch_pairs_text.append((batch[i]['text']['pairs_text']))
        batch_pairs_mask.append((batch[i]['text']['pairs_mask']))
        batch_pairs_segment.append((batch[i]['text']['pairs_segment']))

    batch_right, _ = my_collate_fn_single_hand(batch_right)
    batch_left, _ = my_collate_fn_single_hand(batch_left)
    assert len(batch_left) == len(batch_right)

    batch_body = my_collate_fn_body_pose(batch_body)

    right_pose = []
    right_clips_start = []

    for i in range(len(batch_right)):
        right_pose.append(batch_right[i]['pose_sample'])
        right_clips_start.append(batch_right[i]['clips_start'])

    right_pose = torch.stack(right_pose, dim=0).float()
    right_clips_start = torch.stack(right_clips_start, dim=0).long()

    left_pose = []
    left_clips_start = []

    for i in range(len(batch_left)):
        left_pose.append(batch_left[i]['pose_sample'])
        left_clips_start.append(batch_left[i]['clips_start'])

    left_pose = torch.stack(left_pose, dim=0).float()
    left_clips_start = torch.stack(left_clips_start, dim=0).long()

    body_pose = []
    body_mask = []
    body_clips_start = []

    for i in range(len(batch_body)):
        body_pose.append(batch_body[i]['pose_sample'])
        body_clips_start.append(batch_body[i]['clips_start'])
        body_mask.append(batch_body[i]['pose_mask'])

    body_pose = torch.stack(body_pose, dim=0).float()
    body_clips_start = torch.stack(body_clips_start, dim=0).long()
    body_mask = torch.stack(body_mask, dim=0).long()

    RGB_feature = torch.stack(batch_RGB, dim=0).float()
    pairs_text = torch.stack(batch_pairs_text, dim=0).long()
    pairs_mask = torch.stack(batch_pairs_mask, dim=0).long()
    pairs_segment = torch.stack(batch_pairs_segment, dim=0).long()

    return {'right_pose': right_pose, 'right_clips_start':right_clips_start,
            'left_pose': left_pose, 'left_clips_start':left_clips_start,
            'body_pose': body_pose,  'body_mask': body_mask, 'body_clips_start':body_clips_start,
            'RGB_feature': RGB_feature,
            'pairs_text':pairs_text, 'pairs_mask':pairs_mask, 'pairs_segment':pairs_segment,}