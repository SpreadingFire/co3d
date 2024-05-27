# 计算并校验下载的数据集文件的SHA256哈希值，来保证数据集文件的完整性和正确性
import os
import glob
import argparse
import hashlib
import json

from typing import Optional
from multiprocessing import Pool
from tqdm import tqdm

# 定义默认的哈希值文件路径

# 使用rsplit，从__file__字符串的尾部开始分割，最多分割两次，取第一个部分
DEFAULT_SHA256S_FILE = os.path.join(__file__.rsplit(os.sep, 2)[0], "co3d_sha256.json")

# 每次读取的文件块的大小
# 提高哈希计算的效率
BLOCKSIZE = 65536

# 定一个主函数
def main(
    download_folder: str,

    # 安全哈希算法（SHA），哈希值长度256位
    sha256s_file: str,
    dump: bool = False,

    # 并行处理的线程数量
    n_sha256_workers: int = 4,
    single_sequence_subset: bool = False,
):
    # 检查哈希值文件是否存在
    if not os.path.isfile(sha256s_file):
        raise ValueError(f"The SHA256 file does not exist ({sha256s_file}).")
    
    # 获取哈希值
    expected_sha256s = get_expected_sha256s(
        sha256s_file=sha256s_file,
        single_sequence_subset=single_sequence_subset,
    )

    # 找到所有的zip文件
    zipfiles = sorted(glob.glob(os.path.join(download_folder, "*.zip")))
    print(f"Extracting SHA256 hashes for {len(zipfiles)} files in {download_folder}.")
    extracted_sha256s_list = []
    # 使用多进程计算哈希值
    with Pool(processes=n_sha256_workers) as sha_pool:
        for extracted_hash in tqdm(
            sha_pool.imap(_sha256_file_and_print, zipfiles),
            total=len(zipfiles),
        ):
            extracted_sha256s_list.append(extracted_hash)
            pass

    extracted_sha256s = dict(
        zip([os.path.split(z)[-1] for z in zipfiles], extracted_sha256s_list)
    )

    # 检查并保存哈希值
    if dump:
        print(extracted_sha256s)
        with open(sha256s_file, "w") as f:
            json.dump(extracted_sha256s, f, indent=2)

    missing_keys, invalid_keys = [], []
    # 检查文件的哈希值是否匹配
    for k in expected_sha256s.keys():
        if k not in extracted_sha256s:
            print(f"{k} missing!")
            missing_keys.append(k)
        elif expected_sha256s[k] != extracted_sha256s[k]:
            print(
                f"'{k}' does not match!"
                + f" ({expected_sha256s[k]} != {extracted_sha256s[k]})"
            )
            invalid_keys.append(k)
    if len(invalid_keys) + len(missing_keys) > 0:
        raise ValueError(
            f"Checksum checker failed!"
            + f" Non-matching checksums: {str(invalid_keys)};"
            + f" missing files: {str(missing_keys)}."
        )

# 从哈希文件中获取哈希值
def get_expected_sha256s(
    sha256s_file: str,
    single_sequence_subset: bool = False,
):
    with open(sha256s_file, "r") as f:
        expected_sha256s = json.load(f)
    if single_sequence_subset:
        return expected_sha256s["singlesequence"]
    else:
        return expected_sha256s["full"]

# 检查一个文件的哈希值
def check_co3d_sha256(
    path: str,
    sha256s_file: str,
    expected_sha256s: Optional[dict] = None,
    single_sequence_subset: bool = False,
    do_assertion: bool = True,
):
    zipname = os.path.split(path)[-1]
    if expected_sha256s is None:
        expected_sha256s = get_expected_sha256s(
            sha256s_file=sha256s_file,
            single_sequence_subset=single_sequence_subset,
        )
    extracted_hash = sha256_file(path)
    if do_assertion:
        assert (
            extracted_hash == expected_sha256s[zipname]
        ), f"{zipname}: ({extracted_hash} != {expected_sha256s[zipname]})"
    else:
        return extracted_hash == expected_sha256s[zipname]

# 计算一个文件的哈希值
def sha256_file(path: str):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        file_buffer = f.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            sha256_hash.update(file_buffer)
            file_buffer = f.read(BLOCKSIZE)
    digest_ = sha256_hash.hexdigest()
    # print(f"{digest_} {path}")
    return digest_

# 计算和打印一个文件的哈希值
def _sha256_file_and_print(path: str):
    digest_ = sha256_file(path)
    print(f"{path}: {digest_}")
    return digest_

# 接收并处理用户的输入
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check SHA256 hashes of the CO3D dataset."
    )
    parser.add_argument(
        "--download_folder",
        type=str,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--sha256s_file",
        type=str,
        help="A local target folder for downloading the the dataset files.",
        default=DEFAULT_SHA256S_FILE,
    )
    # 定义一些用户可以设置的参数
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of sha256 extraction workers.",
    )
    parser.add_argument(
        "--dump_sha256s",
        action="store_true",
        help="Store sha256s hashes.",
    )
    parser.add_argument(
        "--single_sequence_subset",
        action="store_true",
        default=False,
        help="Check the single-sequence subset of the dataset.",
    )

    args = parser.parse_args()
    main(
        str(args.download_folder),
        dump=bool(args.dump_sha256s),
        n_sha256_workers=int(args.num_workers),
        single_sequence_subset=bool(args.single_sequence_subset),
        sha256s_file=str(args.sha256s_file),
    ) 
