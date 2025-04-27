import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader_csl_retrieval_train_pose import csl_DataLoader_train_pose, csl_train_pose_collate_fn
from dataloaders.dataloader_csl_retrieval_pose import csl_DataLoader_pose, csl_pose_collate_fn
from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
from dataloaders.dataloader_H2_retrieval_train_pose import H2_DataLoader_train_pose, H2_train_pose_collate_fn
from dataloaders.dataloader_H2_retrieval_pose import H2_DataLoader_pose, H2_pose_collate_fn

def dataloader_csl_train_pose(args, tokenizer):
    train_dataset = csl_DataLoader_train_pose(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        collate_fn=csl_train_pose_collate_fn
    )

    return dataloader, len(train_dataset), sampler


def dataloader_csl_test_pose(args, tokenizer, subset="test"):
    testset = csl_DataLoader_pose(
        subset="test",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )
    dataloader= DataLoader(
        testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=csl_pose_collate_fn,
    )
    return dataloader, len(testset)


def dataloader_ph_train_pose(args, tokenizer):
    train_dataset = ph_DataLoader_train_pose(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        collate_fn=ph_train_pose_collate_fn
    )

    return dataloader, len(train_dataset), sampler


def dataloader_ph_test_pose(args, tokenizer, subset="test"):
    testset = ph_DataLoader_pose(
        subset="test",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )
    dataloader= DataLoader(
        testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=ph_pose_collate_fn,
    )
    return dataloader, len(testset)


def dataloader_h2s_train_pose(args, tokenizer):
    train_dataset = H2_DataLoader_train_pose(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        collate_fn=H2_train_pose_collate_fn
    )

    return dataloader, len(train_dataset), sampler


def dataloader_h2s_test_pose(args, tokenizer, subset="test"):
    testset = H2_DataLoader_pose(
        subset="test",
        data_path=args.data_path,
        features_path=args.features_path,
        features_RGB_path=args.features_RGB_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_len=args.feature_len,
        max_length_frames=args.max_length_frames,
        slide_windows=args.slide_windows,
        windows_stride=args.windows_stride,
        args=args
    )
    dataloader= DataLoader(
        testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=H2_pose_collate_fn,
    )
    return dataloader, len(testset)


DATALOADER_DICT = {}
DATALOADER_DICT["csl_pose"] = {"train":dataloader_csl_train_pose, "dev":dataloader_csl_test_pose, "test":dataloader_csl_test_pose}
DATALOADER_DICT["ph_pose"] = {"train":dataloader_ph_train_pose, "dev":dataloader_ph_test_pose, "test":dataloader_ph_test_pose}
DATALOADER_DICT["h2s_pose"] = {"train":dataloader_h2s_train_pose, "dev":dataloader_h2s_test_pose, "test":dataloader_h2s_test_pose}



