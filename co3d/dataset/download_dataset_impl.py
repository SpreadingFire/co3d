"""
这段代码用于下载并解压指定的数据集。以及提供了一些辅助函数。
这些函数一起工作，可以从网络上下载数据，根据文件的url，然后保存到本地。
其中，一些选项可以自定义，比如是否验证下载文件的校验和，是否在解压之后删除原文件等。
"""

import os
import shutil
import requests
import functools
import json
import warnings

from argparse import ArgumentParser
from typing import List, Optional
from multiprocessing import Pool
from tqdm import tqdm

from .check_checksum import check_co3d_sha256


def download_dataset(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_categories: Optional[List[str]] = None,
    checksum_check: bool = False,
    single_sequence_subset: bool = False,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
    sha256s_file: Optional[str] = None,
):
    """
    下载并解压 CO3D 格式的数据集。

    将在'<download_folder>/_in_progress'文件夹中存储正在下载的文件，
    下载完成后，可以安全地删除该文件夹。

    参数详解如下：
    link_list_file: txt文件,包含下载zip文件的链接列表。 
    download_folder: 文件下载的本地目标文件夹。 
    n_download_workers: 下载数据集文件的并行工作线程数量。 
    n_extract_workers: 解压数据集文件的并行工作线程数量。 
    download_categories: 要下载的类别列表。
    checksum_check: 在解压之前，开启下载文件的校验和验证。
    single_sequence_subset: 是否下载完整数据集的单一序列子集。
    clear_archives_after_unpacking: 在解压后删除不必要的已下载的压缩文件。 
    skip_downloaded_archives: 跳过已经下载的压缩文件。 
    """ 

    
    if checksum_check and not sha256s_file:
        raise ValueError(
            "checksum_check is requested but ground-truth SHA256 file not provided!"
        )

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a json"
            " with zip file download links."
            " For CO3Dv2, the file is stored in the co3d github:"
            " https://github.com/facebookresearch/co3d/blob/main/co3d/links.json"
        )

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    # read the link file
    with open(link_list_file, "r") as f:
        links = json.load(f)

    # get the full dataset links or the single-sequence subset links
    links = links["singlesequence"] if single_sequence_subset else links["full"]

    # split to data links and the links containing json metadata
    metadata_links = []
    data_links = []
    for category_name, urls in links.items():
        for url in urls:
            link_name = os.path.split(url)[-1]
            if single_sequence_subset:
                link_name = link_name.replace("_singlesequence", "")
            if category_name.upper() == "METADATA":
                metadata_links.append((link_name, url))
            else:
                data_links.append((category_name, link_name, url))
        
    if download_categories is not None:
        co3d_categories = set(l[0] for l in data_links)
        not_in_co3d = [c for c in download_categories if c not in co3d_categories]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "dataset categories."
            )
        data_links = [(c, ln, l) for c, ln, l in data_links if c in download_categories]

    with Pool(processes=n_download_workers) as download_pool:
        print(f"Downloading {len(metadata_links)} dataset metadata files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_metadata_file, download_folder),
                metadata_links,
            ),
            total=len(metadata_links),
        ):
            pass

        print(f"Downloading {len(data_links)} dataset files ...")
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_category_file,
                    download_folder,
                    checksum_check,
                    single_sequence_subset,
                    sha256s_file,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "The SHA256 checksums did not match for some of the downloaded files:\n"
                + not_ok_links_str + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    metadata_links = [ml for ml in metadata_links if ml[1].endswith(".zip")]
    print(f"Extracting {len(data_links)} dataset files and {len(metadata_links)} metadata files...")
    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                metadata_links + data_links,
            ),
            total=len(metadata_links) + len(data_links),
        ):
            pass

    print("Done")


def build_arg_parser(
    dataset_name: str,
    default_link_list_file: str,
    default_sha256_file: str,
) -> ArgumentParser:

    """
    构建用于下载数据集的参数解析器。

    参数详解如下：
    dataset_name: 数据集名称
    default_link_list_file: 默认下载文件链接列表
    default_sha256_file: 默认的SHA256校验文件
    """

    
    parser = ArgumentParser(description=f"Download the {dataset_name} dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--download_categories",
        type=lambda x: [x_.strip() for x_ in x.split(",")],
        default=None,
        help=f"A comma-separated list of {dataset_name} categories to download."
        + " Example: 'orange,car' will download only oranges and cars",
    )
    parser.add_argument(
        "--link_list_file",
        type=str,
        default=default_link_list_file,
        help=(
            f"The file with html links to the {dataset_name} dataset files."
            + " In most cases the default local file `links.json` should be used."
        ),
    )
    parser.add_argument(
        "--sha256_file",
        type=str,
        default=default_sha256_file,
        help=(
            f"The file with SHA256 hashes of {dataset_name} dataset files."
            + " In most cases the default local file `co3d_sha256.json` should be used."
        ),
    )
    parser.add_argument(
        "--checksum_check",
        action="store_true",
        default=True,
        help="Check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.add_argument(
        "--no_checksum_check",
        action="store_false",
        dest="checksum_check",
        default=False,
        help="Does not check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.set_defaults(checksum_check=True)
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )

    return parser


def _unpack_category_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    """
    解压指定类别的文件

    参数详解如下：
    download_folder: 文件下载的本地目标文件夹
    clear_archive: 是否在解压后删除原压缩文件
    link: 文件链接
    """
    
    *_, link_name, url = link
    local_fl = os.path.join(download_folder, link_name)
    print(f"Unpacking dataset file {local_fl} ({link_name}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    if clear_archive:
        os.remove(local_fl)


def _download_category_file(
    download_folder: str,
    checksum_check: bool,
    single_sequence_subset: bool,
    sha256s_file: Optional[str],
    skip_downloaded_files: bool,
    link: str,
):
    """
    下载指定类别的文件

    参数详解如下：
    download_folder: 文件下载的本地目标文件夹
    checksum_check: 检查SHA256校验和
    single_sequence_subset: 是否下载完整数据集的单一序列子集
    sha256s_file: SHA256校验文件
    skip_downloaded_files: 跳过已经下载的文件
    link: 文件链接
    """
    
    category, link_name, url = link
    local_fl_final = os.path.join(download_folder, link_name)

    if skip_downloaded_files and os.path.isfile(local_fl_final):
        print(f"Skipping {local_fl_final}, already downloaded!")
        return link_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    local_fl = os.path.join(in_progress_folder, link_name)

    print(f"Downloading dataset file {link_name} ({url}) to {local_fl}.")
    _download_with_progress_bar(url, local_fl, link_name)
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        try:
            check_co3d_sha256(
                local_fl,
                sha256s_file=sha256s_file,
                single_sequence_subset=single_sequence_subset,
            )
        except AssertionError:
            warnings.warn(
                f"Checksums for {local_fl} did not match!"
                + " This is likely due to a network failure,"
                + " please restart the download script." 
            )
            return link_name, False
        
    os.rename(local_fl, local_fl_final)
    return link_name, True


def _download_metadata_file(download_folder: str, link: str):
    """
    下载数据集的元数据文件

    参数详解如下：
    download_folder: 文件下载的本地目标文件夹
    link: 文件链接
    """
    local_fl = os.path.join(download_folder, link[0])
    # remove the singlesequence postfix in case we are downloading the s.s. subset
    local_fl = local_fl.replace("_singlesequence", "")
    print(f"Downloading dataset metadata file {link[1]} ({link[0]}) to {local_fl}.")
    _download_with_progress_bar(link[1], local_fl, link[0])


def _download_with_progress_bar(url: str, fname: str, filename: str):
    """
    展示下载进度条的函数

    参数详解如下：
    url: 下载链接
    fname: 文件在本地的保存名称
    filename: 文件名称
    """
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max((max(total // 1024, 1) // 20), 1) == 0:
                print(f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%.")
                print(bar)
