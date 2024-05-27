"""
主要目标：抹除测试集的标签信息

被抹去的信息包括了"fg_probability"（可能是前景概率的意思），
"image_rgb"（RGB色彩空间的图像数据），
"depth_map"（深度图，记录了图像中各像素点的深度信息）和"mask_crop"。

redact_eval_frame_data函数是用来将评估所用的帧数据的测试元素（第一张图片）的所有信息抹去的，
_check_valid_eval_frame_data函数则是用来检查帧数据是否被正确抹去的。
"""

import torch
import copy
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet


def redact_eval_frame_data(fd: FrameData) -> FrameData:
    """
    将评估所用的帧数据 `fd` 的测试元素（第一张图片）的所有信息抹去。

    通过将 `fd` 中的相应张量的所有元素置零并删除 field `sequence_point_cloud` 来实现。
    """
    # 通过深拷贝来创建一个新的 FrameData 对象
    fd_redacted = copy.deepcopy(fd)
    for redact_field_name in [
        "fg_probability",
        "image_rgb",
        "depth_map",
        "mask_crop",
    ]:
        # 将需要抹去的字段的所有元素置零
        field_val = getattr(fd, redact_field_name)
        field_val[:1] *= 0
    # 移除点云信息
    fd_redacted.sequence_point_cloud_idx = None
    fd_redacted.sequence_point_cloud = None
    return fd_redacted # 返回处理后的 FrameData 对象


def _check_valid_eval_frame_data(
    fd: FrameData,
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
):
    """
    检查评估批次 `fd` 是否被正确抹去。
    """
    # 检查被抹去的字段和图片索引关系
    is_redacted = torch.stack(
        [
            getattr(fd, k).abs().sum((1,2,3)) <= 0
            for k in ["image_rgb", "depth_map", "fg_probability"]
        ]
    )
    # 测试集应满足的条件
    if sequence_set==CO3DSequenceSet.TEST:
        # 第一张图片应被抹去
        assert is_redacted[:, 0].all()
        # 所有深度图应被抹去
        assert is_redacted[1, :].all()
        # 已知的视图不应被抹去
        assert not is_redacted[:, 1:].all(dim=0).any()
    # 开发集应满足的条件
    elif sequence_set==CO3DSequenceSet.DEV:
        # 没有信息应被抹去
        assert not is_redacted.all(dim=0).any()
    else:
        raise ValueError(sequence_set) # 其他情况，报错
