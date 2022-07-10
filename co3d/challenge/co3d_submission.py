# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import logging
import errno
import pickle
import glob

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate

from co3d.challenge.metric_utils import EVAL_METRIC_NAMES

from .utils import evaluate_file_folders
from .data_types import RGBDAFrame, CO3DTask, CO3DSequenceSet
from .io import (
    load_all_eval_batches,
    store_rgbda_frame,
)


logger = logging.getLogger(__file__)


class CO3DSubmission:
    def __init__(
        self,
        task: CO3DTask,
        sequence_set: CO3DSequenceSet,
        output_folder: str,
        dataset_root: Optional[str] = None,
        server_data_folder: Optional[str] = None,
        on_server: bool = False,
    ):
        self.task = task
        self.sequence_set = sequence_set
        self.output_folder = output_folder
        self.dataset_root = dataset_root
        self.server_data_folder = server_data_folder
        self.on_server = on_server
        self.submission_archive = os.path.join(
            output_folder, f"submission_{task.value}_{sequence_set.value}.zip"
        )
        self.evaluate_exceptions_file = os.path.join(output_folder, "eval_exceptions.pkl")
        self.submission_cache = os.path.join(output_folder, "submission_cache")
        os.makedirs(self.submission_cache, exist_ok=True)
        self._result_list = []
        self._eval_batches_map = None
    
    @staticmethod
    def get_submission_cache_image_dir(
        output_folder: str,
        category: str,
        subset_name: str,
    ):
        """
        Get the cache folder containing all predictions of a given category frame set.

        Args:
            output_folder: The root submission folder.
            category: CO3D category name (e.g. "apple", "orange")
            subset_name: CO3D subset name (e.g. "manyview_dev_0", "manyview_test_0")
        """
        return os.path.join(output_folder, category, subset_name)

    def add_result(
        self,
        category: str,
        subset_name: str,
        sequence_name: str,
        frame_number: int,
        image: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        """
        Adds a single user-predicted image to the current submission.

        Args:
            category: The CO3D category of the image (e.g. "apple", "car").
            subset_name: The name of the subset which the image comes from
                (e.g. "manyview_dev_0", "manyview_test_0").
            sequence_name: The name of the sequence which the image comes from.
            frame_number: The number of the corresponding ground truth frame.
            image: 3xHxW numpy.ndarray containing the RGB image.
                The color range is [0-1] and `image` should be of the same size
                as the corresponding ground truth image.
            mask: `1xHxW numpy.ndarray containing the binary foreground mask of the
                rendered object.
                The values should be in {0, 1} and `mask` should be of the same size
                as the corresponding ground truth image.
            depth: `1xHxW numpy.ndarray containing the rendered depth map of the predicted
                image.
                The depth map should be of the same size as the corresponding
                ground truth image.
        """
        res = CO3DSubmissionRender(
            category=category,
            subset_name=subset_name,
            sequence_name=sequence_name,
            frame_number=frame_number,
            rgbda_frame=None,
        )
        res_file = res.get_image_path(self.submission_cache)
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        logger.debug(f"Storing submission files {res_file}.")
        store_rgbda_frame(
            RGBDAFrame(image=image, mask=mask, depth=depth),
            res_file,
        )
        self._result_list.append(res)

    def _get_result_frame_index(self):
        return {(res.sequence_name, res.frame_number): res for res in self._result_list}

    def get_eval_batches_map(self):
        """
        Returns a dictionary of evaluation examples of the following form:
            ```
            {(category: str, subset_name: str): eval_batches}  # eval_batches_map
            ```
        where `eval_batches` look as follows:
            ```
            [
                [
                    (sequence_name_0: str, frame_number_0: int),
                    (sequence_name_0: str, frame_number_1: int),
                    ...
                    (sequence_name_0: str, frame_number_M: int),
                ],
                ...
                [
                    (sequence_name_N: str, frame_number_0: int),
                    (sequence_name_N: str, frame_number_1: int),
                    ...
                    (sequence_name_N: str, frame_number_M: int),
                ]
            ]  # eval_batches
            ```
        Here, `eval_batches' containing a list of `N` evaluation examples, 
        each consisting of a tuple of frames with numbers `frame_number_j`
        from a given sequence name `sequence_name_i`.

        Note that the mapping between `frame_number` and `sequence_name` to the CO3D
        data is stored in the respective `frame_annotations.jgz` and `sequence_annotation.jgz`
        files in `<dataset_root>/<category>`.

        Returns:
            eval_batches_map: A dictionary of evaluation examples for each category.
        """
        if self._eval_batches_map is None:
            self._eval_batches_map = load_all_eval_batches(
                self.dataset_root,
                self.task,
                self.sequence_set,
                remove_frame_paths=False,
            )    
        return self._eval_batches_map

    def clear_files(self):
        """
        Remove all generated submission files.
        """
        if os.path.isdir(self.output_folder):
            shutil.rmtree(self.output_folder)
        if os.path.isdir(self.submission_cache):
            shutil.rmtree(self.submission_cache)
        if os.path.isfile(self.submission_archive):
            os.remove(self.submission_archive)

    def validate_export_results(self):
        if self.dataset_root is None or not os.path.isdir(self.dataset_root):
            raise ValueError(
                "For validating the results, dataset_root has to be defined"
                + " and has to point to a valid root folder of the CO3D dataset."
            )
        eval_batches_map = self.get_eval_batches_map()
        result_frame_index = self._get_result_frame_index()
        valid = True
        for (category, subset_name), eval_batches in eval_batches_map.items():
            eval_batches_2tuple = [tuple(b[:2]) for b in eval_batches]
            
            missing_preds = [
                b for b in eval_batches_2tuple if b not in result_frame_index
            ]
            if len(missing_preds) > 0:
                valid = False
                logger.info(
                    f"{category}/{subset_name} is missing predictions."
                )
                logger.debug(str(missing_preds))
                
            additional_results = [
                idx for idx, res in result_frame_index.items() if (
                    idx not in eval_batches_2tuple
                    and res.category==category and res.subset_name==subset_name
                )
            ]
            if len(additional_results) > 0:
                valid = False
                logger.info(
                    f"{category}/{subset_name} has additional results."
                )
                logger.debug(str(additional_results))
                
        return valid

    def export_results(self, validate_results: bool = True):
        """
        Export the generated evaluation images for a submission to the EvalAI server.

        Args:
            validate_results: If `True`, checks whether the added results are valid
                before submission. This requires setting `self.dataset_root` to a directory
                containing a local copy of the CO3D dataset.
        """
        if validate_results:
            # optionally check that all results are correct
            valid_results = self.validate_export_results()
            if not valid_results:
                logger.warning(
                    "The submission results are invalid."
                    " The evaluation will be incomplete."
                )
        
        # First we need to remove all links to the ground truth directories
        # that were potentially created during a call to self.evaluate().
        self._clear_gt_links()

        # zip the directory
        shutil.make_archive(
            base_name=self.submission_archive.replace(".zip", ""),
            format="zip",
            root_dir=self.submission_cache,
            base_dir=self.submission_cache,
        )

        # finally export the result
        logger.warning(
            f"Exported result file: \n\n    ===> {self.submission_archive} <==="
            f"\n\nYou can now submit the file to the EvalAI server"
            f" ('{self.task.value}' track)."
        )

    def _clear_gt_links(self):
        gt_folders = glob.glob(os.path.join(self.submission_cache, "*", "GT_*"))
        for gt_folder in gt_folders:
            logger.debug(f"Clearing GT link directory {gt_folder}.")
            shutil.rmtree(gt_folder)


    def evaluate_zip_file(self, zip_path: str):
        os.makedirs(self.submission_cache, exist_ok=True)
        shutil.unpack_archive(zip_path, self.submission_cache, "zip")
        return self.evaluate()


    def evaluate(self):
        if self.on_server:
            if not os.path.isdir(self.server_data_folder):
                raise ValueError(
                    "For evaluation on the server server_data_folder has to be specified."
                )
        else:
            if not os.path.isdir(self.dataset_root):
                raise ValueError("For evaluation dataset_root has to be specified.")
            if self.sequence_set == CO3DSequenceSet.TEST:
                raise ValueError("Cannot evaluate on the hidden test set!")

        eval_batches_map = self.get_eval_batches_map()

        # buffers for results and exceptions
        eval_exceptions = {}
        eval_results = {}

        for (category, subset_name), eval_batches in eval_batches_map.items():
            logger.info(f"Evaluating {category}/{subset_name}.")

            pred_category_subset_dir = CO3DSubmission.get_submission_cache_image_dir(
                self.submission_cache,
                category,
                subset_name,
            )

            # The case with no predicted results.
            if (
                (not os.path.isdir(pred_category_subset_dir))
                or (len(os.listdir(pred_category_subset_dir))==0)
            ):
                logger.info(f"No evaluation predictions for {category}/{subset_name}")
                eval_results[(category, subset_name)] = (None, None)
                eval_exceptions[(category, subset_name)] = None
                continue

            # Make a temporary GT folder with symlinks to GT data based on eval batches
            gt_category_subset_dir = CO3DSubmission.get_submission_cache_image_dir(
                self.submission_cache,
                category,
                "GT_" + subset_name,
            )
            
            for b in eval_batches:
                if self.on_server:
                    _link_eval_batch_data_from_server_db_to_gt_tempdir(
                        self.server_data_folder,
                        gt_category_subset_dir,
                        category,
                        b,
                    )
                else:
                    _link_eval_batch_data_from_dataset_root_to_gt_tempdir(
                        self.dataset_root,
                        gt_category_subset_dir,
                        category,
                        b,
                    )

            # Evaluate and catch any exceptions.
            try:
                eval_results[(category, subset_name)] = evaluate_file_folders(
                    pred_category_subset_dir,
                    gt_category_subset_dir,
                )
            except Exception as exc:
                logger.warning(f"Evaluation of {category}/{subset_name} failed!", exc_info=True)
                eval_results[(category, subset_name)] = (None, None)
                eval_exceptions[(category, subset_name)] = exc

        # Get the average results.
        average_results = {}
        for m in EVAL_METRIC_NAMES:
            # Automatically generates NaN average if some results are missing.
            average_results[m] = sum(
                eval_result[m] if eval_result is not None else float("NaN")
                for eval_result, _ in eval_results.values()
            ) / len(eval_results)
        eval_results[("MEAN", "-")] = average_results, None

        # Generate a nice table and print.
        tab_rows = []
        for (category, subset_name), (eval_result, _) in eval_results.items():
            tab_row = [category, subset_name]
            if eval_result is None:
                tab_row.extend([float("NaN")] * len(EVAL_METRIC_NAMES))
            else:
                tab_row.extend([eval_result[k] for k in EVAL_METRIC_NAMES])
            tab_rows.append(tab_row)

        table_str = tabulate(tab_rows, headers=["Category", "Subset name", *EVAL_METRIC_NAMES])
        logger.info("\n"+table_str)

        # Store the human-readable table
        table_txt_file = os.path.join(self.output_folder, "results.txt")
        logger.info(f"Dumping the results table to {table_txt_file}.")
        with open(table_txt_file, 'w') as f:
            f.write(table_str)

        # Store the recorded exceptions in the submissions folder.
        with open(self.evaluate_exceptions_file, "wb") as f:
            pickle.dump(eval_exceptions, f)

        return eval_results


@dataclass
class CO3DSubmissionRender:
    """
    Contains information about a single predicted image.

    category: The name of the category of the prediction.
    subset_name: The dataset subset of the prediction.
    frame_number: The number of the corresponding ground truth frame.
    rgbda_frame: The actual render.
    """
    category: str
    subset_name: str
    sequence_name: str
    frame_number: int
    rgbda_frame: Optional[RGBDAFrame] = None

    def get_image_path(self, root_dir: str):
        return os.path.join(
            CO3DSubmission.get_submission_cache_image_dir(
                root_dir,
                self.category,
                self.subset_name,
            ),
            self.get_image_name(),
        )

    def get_image_name(self):
        return get_submission_image_name(
            self.category, self.sequence_name, self.frame_number
        )


def get_submission_image_name(category: str, sequence_name: str, frame_number: str):
    return f"{category}_{sequence_name}_{frame_number}"


def _link_eval_batch_data_from_dataset_root_to_gt_tempdir(
    dataset_root: str,
    temp_dir: str,
    category: str,
    frame_index: Tuple[str, int, str],
):
    sequence_name, frame_number, gt_image_path = frame_index
    image_name = get_submission_image_name(category, sequence_name, frame_number)
    os.makedirs(temp_dir, exist_ok=True)
    for data_type in ["image", "depth", "mask", "depth_mask"]:
        gt_data_path = gt_image_path.replace("/images/", f"/{data_type}s/")
        if data_type=="depth":
            gt_data_path = gt_data_path.replace(".jpg", ".jpg.geometric.png")
        elif data_type in ("mask", "depth_mask"):
            gt_data_path = gt_data_path.replace(".jpg", ".png")
        tgt_image_name = f"{image_name}_{data_type}.png"
        src = os.path.join(dataset_root, gt_data_path)
        dst = os.path.join(temp_dir, tgt_image_name)
        logger.debug(f"{src} <--- {dst}")
        _symlink_force(src, dst)


def _link_eval_batch_data_from_server_db_to_gt_tempdir(
    server_folder: str,
    temp_dir: str,
    category: str,
    frame_index: Tuple[str, int, str],
):
    sequence_name, frame_number, _ = frame_index
    image_name = get_submission_image_name(category, sequence_name, frame_number)
    os.makedirs(temp_dir, exist_ok=True)
    for data_type in ["image", "depth", "mask", "depth_mask"]:
        image_name_postfixed = image_name + f"_{data_type}.png"
        src = os.path.join(server_folder, image_name_postfixed)
        dst = os.path.join(temp_dir, image_name_postfixed)
        logger.debug(f"{src}<---{dst}")
        _symlink_force(src, dst)


def _symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e