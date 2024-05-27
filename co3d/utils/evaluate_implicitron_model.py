# 导入所需的库和模块
import logging  # 用于记录日志
import os  # 用于操作系统相关的功能
import torch  # PyTorch库，用于深度学习
import json  # 用于处理JSON格式的数据
import warnings  # 用于处理警告
from typing import Optional, Union, Dict, Tuple  # 用于类型提示
from tqdm import tqdm  # 用于显示进度条
from omegaconf import DictConfig, OmegaConf  # 用于处理配置文件
import numpy as np  # 数值计算库

import pytorch3d  # PyTorch3D库，用于3D计算
from pytorch3d.implicitron.models.generic_model import ImplicitronRender, GenericModel  # Implicitron模型
from pytorch3d.implicitron.tools.config import get_default_args  # 获取默认配置
from pytorch3d.implicitron.dataset.dataset_base import FrameData  # 数据集基类
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap  # 数据集映射提供者
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2  # JSON索引数据集映射提供者
)
from pytorch3d.implicitron.tools.config import expand_args_fields  # 扩展参数字段
from pytorch3d.implicitron.tools.model_io import (
    parse_epoch_from_model_path,  # 从模型路径解析epoch
    find_last_checkpoint  # 找到最新的检查点
)
from pytorch3d.implicitron.models.renderer.base import (
    EvaluationMode  # 评估模式
)

from co3d.utils import dbir_utils  # 导入CO3D的工具
from co3d.challenge.co3d_submission import CO3DSubmission  # 导入CO3D提交类
from co3d.challenge.data_types import CO3DTask, CO3DSequenceSet  # 导入CO3D任务和序列集类型
from co3d.challenge.utils import (
    get_co3d_task_from_subset_name,  # 从子集名称获取CO3D任务
    get_co3d_sequence_set_from_subset_name  # 从子集名称获取CO3D序列集
)
from co3d.dataset.utils import redact_eval_frame_data, _check_valid_eval_frame_data  # 数据集实用工具
from co3d.challenge.metric_utils import EVAL_METRIC_NAMES  # 评估指标名称

# 获取数据集根目录
DATASET_ROOT = os.getenv("CO3DV2_DATASET_ROOT")
DATASET_ROOT_HIDDEN = os.getenv("CO3DV2_HIDDEN_DATASET_ROOT")

# 强制将implicitron_trainer添加到路径中
_pytorch3d_root = os.path.dirname(os.path.dirname(pytorch3d.__file__))
implicitron_trainer_dir = os.path.join(_pytorch3d_root, "projects", "implicitron_trainer")
from projects.implicitron_trainer.experiment import Experiment  # 导入实验类

logger = logging.getLogger(__name__)  # 获取日志记录器

def evaluate_implicitron_exp_dir_map(
    category_subset_implicitron_exp_dirs: Union[Dict[Tuple[str, str], str], str],
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
    submission_output_folder: str,
    num_eval_workers: int = 4,
    submit_to_eval_ai: bool = False,
    skip_evaluation: bool = False,
    fill_results_from_cache: bool = False,
    implicitron_exp_dir_submission_output_subfolder: Optional[str] = None
):
    """
    评估并提交到EvalAI，评估以下两种情况：
        1) 所有特定类别的Implicitron模型
        2) 用于所有类别的单一模型

    参数：
        category_subset_implicitron_exp_dirs: 两种选择：
            1) 一个字典，包含从每个CO3Dv2类别和子集到相应Implicitron模型实验目录的映射。
            2) 一个字符串，包含用于重建所有类别的单一模型路径。
        task: CO3D任务 - CO3DTask.MANY_VIEW或CO3DTask.FEW_VIEW。
        sequence_set: 要评估的序列集：CO3DSequenceSet.DEV用于开发集，CO3DSequenceSet.TEST用于测试集。
        submission_output_folder: 包含提交输出文件的目录。
        num_eval_workers: 执行评估的进程数。
        submit_to_eval_ai: 如果为True，将自动使用CLI界面将导出的结果归档提交到EvalAI。
        skip_evaluation: 跳过本地评估。
        implicitron_exp_dir_submission_output_subfolder: 如果设置为字符串，将从预计算结果加载。
    """
    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=submission_output_folder,
        dataset_root=DATASET_ROOT,
    )

    if fill_results_from_cache:
        submission.fill_results_from_cache()
    else:
        if not isinstance(category_subset_implicitron_exp_dirs, str):
            for category, subset_name in submission.get_eval_batches_map():
                if (category, subset_name) not in category_subset_implicitron_exp_dirs:
                    raise ValueError(
                        f"缺少{category}/{subset_name}的Implicitron实验目录。"
                    )

        for category, subset_name in submission.get_eval_batches_map():
            if isinstance(category_subset_implicitron_exp_dirs, str):
                current_implicitron_exp_dir = category_subset_implicitron_exp_dirs
            else:
                current_implicitron_exp_dir = category_subset_implicitron_exp_dirs[
                    (category, subset_name)
                ]

            if implicitron_exp_dir_submission_output_subfolder is not None:
                submission.link_results_from_existing_output_folder(
                    os.path.join(
                        current_implicitron_exp_dir,
                        implicitron_exp_dir_submission_output_subfolder
                    )
                )
            else:
                update_implicitron_submission_with_category_and_subset_predictions(
                    submission=submission,
                    implicitron_exp_dir=current_implicitron_exp_dir,
                    dataset_root=DATASET_ROOT,
                    category=category,
                    subset_name=subset_name,
                    n_known_frames_for_test=9 if task == CO3DTask.MANY_VIEW else 0,
                )

    if sequence_set != CO3DSequenceSet.TEST and not skip_evaluation:
        submission.evaluate(num_workers=num_eval_workers)
    
    if submit_to_eval_ai:
        submission.export_results(validate_results=True)
        submission.submit_to_eval_ai()


def evaluate_implicitron_exp_dir(
    implicitron_exp_dir: str,
    task: Optional[CO3DTask] = None,
    sequence_set: Optional[CO3DSequenceSet] = None,
    subset_name: Optional[str] = None,
    category: Optional[str] = None,
    result_dump_file: Optional[str] = None,
    clear_submission_cache_before_evaluation: bool = False,
    clear_submission_cache_after_evaluation: bool = False,
    submission_output_folder: Optional[str] = None,
    num_eval_workers: int = 4
):
    """
    运行Implicitron实验目录的评估。
    除非用户覆盖，否则此函数会自动解析类别/子集/任务/序列集/数据集根目录。

    参数：
        implicitron_exp_dir: Implicitron实验目录。
        task: CO3D任务 - CO3DTask.MANY_VIEW或CO3DTask.FEW_VIEW。
        sequence_set: 要评估的序列集：CO3DSequenceSet.DEV用于开发集，CO3DSequenceSet.TEST用于测试集。
        subset_name: CO3Dv2子集的名称。
        category: CO3Dv2类别的名称。
        result_dump_file: 评估结果的JSON文件路径。
        clear_submission_cache_before_evaluation: 在开始当前评估运行之前删除所有以前的中间提交文件。
        clear_submission_cache_after_evaluation: 在评估运行后删除所有中间提交文件。
        submission_output_folder: 中间提交文件的路径。
        num_eval_workers: 执行评估的进程数。
    """
    if result_dump_file is None:
        result_dump_file = os.path.join(implicitron_exp_dir, "results_challenge_eval.json")

    cfg = load_implicitron_config_from_exp_dir(implicitron_exp_dir)  

    assert (
        cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_class_type
        == "JsonIndexDatasetMapProviderV2"
    )

    dataset_provider_args = (
        cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_JsonIndexDatasetMapProviderV2_args
    )
    if subset_name is None:
        subset_name = dataset_provider_args.subset_name
    if category is None:
        category = dataset_provider_args.category
    if task is None:
        task = get_co3d_task_from_subset_name(subset_name)
    if sequence_set is None:
        sequence_set = get_co3d_sequence_set_from_subset_name(subset_name)
    
    dataset_root = (
        DATASET_ROOT
        if DATASET_ROOT is not None
        else dataset_provider_args.dataset_root
    )

    logger.info(
        f"Evaluating Implicitron model on category {category}; subset {subset_name}"
    )

    if submission_output_folder is None:
        submission_output_folder = get_default_implicitron_exp_dir_submission_output_folder(
            implicitron_exp_dir=implicitron_exp_dir
        )

    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=submission_output_folder,
        dataset_root=dataset_root,
    )

    if clear_submission_cache_before_evaluation:
        submission.clear_submission_cache()
    
    update_implicitron_submission_with_category_and_subset_predictions(
        submission=submission,
        implicitron_exp_dir=implicitron_exp_dir,
        dataset_root=dataset_root,
        category=category,
        subset_name=subset_name,
        n_known_frames_for_test=9 if task == CO3DTask.MANY_VIEW else 0,
    )

    submission.evaluate(num_workers=num_eval_workers)
    submission.export_results(result_dump_file=result_dump_file, validate_results=True)

    if clear_submission_cache_after_evaluation:
        submission.clear_submission_cache()


def get_default_implicitron_exp_dir_submission_output_folder(
    implicitron_exp_dir: str
) -> str:
    """
    返回用于中间提交文件的默认输出目录。
    """
    return os.path.join(implicitron_exp_dir, "challenge_submission")


def load_implicitron_config_from_exp_dir(
    implicitron_exp_dir: str
) -> DictConfig:
    """
    从给定的Implicitron实验目录加载配置文件。
    """
    cfg_file = os.path.join(implicitron_exp_dir, "config.yaml")
    assert os.path.isfile(cfg_file), f"找不到配置文件{cfg_file}"
    return OmegaConf.load(cfg_file)


def update_implicitron_submission_with_category_and_subset_predictions(
    submission: CO3DSubmission,
    implicitron_exp_dir: str,
    dataset_root: str,
    category: str,
    subset_name: str,
    n_known_frames_for_test: int = 0,
):
    """
    更新提交对象以包含Implicitron模型的预测结果。

    参数：
        submission: CO3DSubmission对象。
        implicitron_exp_dir: Implicitron实验目录。
        dataset_root: 数据集根目录。
        category: CO3Dv2类别的名称。
        subset_name: CO3Dv2子集的名称。
        n_known_frames_for_test: 测试时已知的帧数。
    """

    logger.info(
        "Runing depth-based image rendering (DBIR) new view synthesis "
        f"on category '{category}' subset '{subset_name}'"
    )

    # Get the evaluation device.
    device = torch.device("cuda") if torch.cuda.is_available() else device("cpu")

    # load the implicitron model
    model = load_model_from_implicitron_exp_dir(implicitron_exp_dir)

    # Determine the sequence set and the task we are solving
    sequence_set = submission.sequence_set
    task = submission.task

    # Obtain the CO3Dv2 dataset map
    dataset_map = get_dataset_map(
        dataset_root,
        category,
        subset_name,
        n_known_frames_for_test=n_known_frames_for_test,
    )

    # The test dataloader simply iterates over test_dataset.eval_batches
    # this is done by setting test_dataset.eval_batches as the batch sampler
    test_dataset = dataset_map["test"]
    eval_batches = test_dataset.get_eval_batches()

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=eval_batches,
        num_workers=num_workers,
        collate_fn=FrameData.collate,
    )

    # loop over eval examples
    logger.info(
        f"Rendering {len(test_dataloader)} test views for {category}/{subset_name}"
    )

    if sequence_set==CO3DSequenceSet.TEST:
        # the test set contains images with redacted foreground masks which cause
        # the test dataloader to spam a warning message,
        # we suppress this warning with the following line
        warnings.filterwarnings("ignore", message="Empty masks_for_bbox.*")
    
    for eval_index, eval_frame_data in enumerate(tqdm(test_dataloader)):
        # the first element of eval_frame_data is the actual evaluation image,
        # the 2nd-to-last elements are the knwon source images used for building 
        # the reconstruction (source images are present only for the few-view task)

        # move the eval data to the requested device
        eval_frame_data = eval_frame_data.to(device)

        # sanity check that the eval frame data has correctly redacted entries
        _check_valid_eval_frame_data(eval_frame_data, task, sequence_set)

        # Redact the frame data so we are sure we cannot use the data
        # from the actual unobserved evaluation sample
        eval_frame_data = redact_eval_frame_data(eval_frame_data)

        # Obtain the image render. In case dataset_test.box_crop==True,
        # we need to paste the render back to the original image bounds.
        model_preds = model(
            **eval_frame_data,
            eval_mode=EvaluationMode.EVALUATION,
        )
        render_crop = model_preds["implicitron_render"]

        # cut the valid part of the render and paste into the original image canvas
        render_full_image = dbir_utils.paste_render_to_original_image(
            eval_frame_data, render_crop
        )

        # get the image, mask, depth as numpy arrays for the challenge submission
        image, mask, depth = [
            getattr(render_full_image, f"{data_type}_render").cpu().numpy()[0]
            for data_type in ["image", "mask", "depth"]
        ]

        # clip the rendered image to [0, 1] range
        image = image.clip(0.0, 1.0)

        # add the results to the submission object
        submission.add_result(
            category=category,
            subset_name=subset_name,
            sequence_name=eval_frame_data.sequence_name[0],
            frame_number=int(eval_frame_data.frame_number[0]),
            image=image,
            mask=mask,
            depth=depth,
        )

    # reset all warnings
    warnings.simplefilter("always")


def get_default_implicitron_exp_dir_submission_output_folder(
    implicitron_exp_dir: str,
    task: CO3DTask,
    sequence_set: CO3DSequenceSet,
):
    return os.path.join(
        implicitron_exp_dir,
        f"implicitron_submission_output_{task.value}_{sequence_set.value}",
    )


def parse_co3d_challenge_settings_from_implicitron_exp_dir(
    implicitron_exp_dir: str
) -> Tuple[CO3DSequenceSet, CO3DTask, str, str]:
    """
    Reads the config of an implicitron experiment stored in `implicitron_exp_dir` and
    returns the configuration of the corresponding challenge entry.

    Args:
        implicitron_exp_dir: The directory of an Implicitron experiment.
    Returns:
        sequence_set: CO3D sequence set of the experiment.
        task: The CO3D task of the experiment.
        category: The category of the experiment.
        subset_name: The name of the CO3D subset.
    """

    cfg = load_implicitron_config_from_exp_dir(implicitron_exp_dir)
    dataset_provider_args = (
        cfg
        .data_source_ImplicitronDataSource_args
        .dataset_map_provider_JsonIndexDatasetMapProviderV2_args
    )
    subset_name = dataset_provider_args.subset_name
    category = dataset_provider_args.category
    task = get_co3d_task_from_subset_name(subset_name)
    sequence_set = get_co3d_sequence_set_from_subset_name(subset_name)
    return sequence_set, task, category, subset_name


def load_implicitron_config_from_exp_dir(implicitron_exp_dir: str):
    cfg_filename = os.path.join(implicitron_exp_dir, "expconfig.yaml")
    cfg_load = OmegaConf.load(cfg_filename)
    cfg_default = get_default_args(Experiment)
    cfg = OmegaConf.merge(cfg_default, cfg_load)
    cfg.exp_dir = implicitron_exp_dir
    return cfg


def load_model_from_implicitron_exp_dir(exp_dir: str) -> GenericModel:
    cfg = load_implicitron_config_from_exp_dir(exp_dir)
    experiment = Experiment(**cfg)
    experiment.model_factory.force_resume = True
    model = experiment.model_factory(accelerator=None, exp_dir=exp_dir)
    model.cuda()
    model.eval()
    return model


def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
    n_known_frames_for_test: int = 0,
) -> DatasetMap:
    """
    Obtain the dataset map that contains the train/val/test dataset objects.
    """
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map_provider = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig({"remove_empty_masks": False}),
        n_known_frames_for_test=n_known_frames_for_test,
    )
    return dataset_map_provider.get_dataset_map()


def print_category_subset_results(category_subset_results: Dict[str, float]):
    for k, v in category_subset_results.items():
        print(f"{k:20s}: {v:1.3f}")
