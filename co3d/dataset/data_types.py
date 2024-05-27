 """
代码目的
    定义数据结构：代码使用Python的dataclass模块定义了多种数据结构，这些数据结构用于表示图像、深度、遮罩、视点等注解信息。
    数据的序列化和反序列化：提供了函数将这些数据类对象序列化为JSON格式（包括压缩格式），以及从JSON格式反序列化回数据类对象。
                          这对于数据的存储和加载非常有用。

  """



import sys
import dataclasses
import gzip
import json
from dataclasses import dataclass, Field, MISSING
from typing import Any, cast, Dict, IO, Optional, Tuple, Type, TypeVar, Union

import numpy as np

# 根据Python版本导入相应的模块
if sys.version_info >= (3, 8, 0):
    from typing import get_args, get_origin
elif sys.version_info >= (3, 7, 0):
    def get_origin(cls):  # pragma: no cover
        return getattr(cls, "__origin__", None)

    def get_args(cls):  # pragma: no cover
        return getattr(cls, "__args__", None)
else:
    raise ImportError("本模块需要Python 3.7+版本")

_X = TypeVar("_X")
TF3 = Tuple[float, float, float]

# 定义图像注解的数据类
@dataclass
class ImageAnnotation:
    path: str  # 相对于数据集根目录的jpg文件路径
    size: Tuple[int, int]  # 图像的高度和宽度

# 定义深度注解的数据类
@dataclass
class DepthAnnotation:
    path: str  # 相对于数据集根目录的png文件路径，存储 'depth/scale_adjustment'
    scale_adjustment: float  # 一个将png值转换为实际深度的因子：‘depth=png*scale_adjustment’
    mask_path: Optional[str]  # 相对于数据集根目录的png文件路径，存储二元‘depth’ mask

# 定义遮罩注解的数据类
@dataclass
class MaskAnnotation:
    path: str  # 存储 (Prob(fg | pixel) * 255) 的png文件路径
    mass: Optional[float] = None  # 遮罩中的像素数量；sum(Prob(fg | pixel))



# 定义视点注解的数据类
 """
它为记录和处理视点（即摄像机或者观察者的视点）相关的信息提供了一种方便的数据结构。
这个数据类具有以下的属性：
"R"：为一个3x3的矩阵，代表了旋转向量。在这里，它将世界坐标系下的坐标（X_world）旋转到摄像机坐标系下的坐标（X_cam），
     即X_cam = X_world @ R + T，这是一个典型的坐标变换公式。
"T"：为一个平移向量，表示了世界坐标系原点到摄像机坐标系原点的位移。
"focal_length"：是一个元组，含有两个浮点数，它们分别表示了摄像机的水平焦距和垂直焦距。
"principal_point"：也是一个元组，含有两个浮点数，它们表示了摄像机的主点在水平和垂直方向上的坐标。
"intrinsics_format"：这是一个字符串，定义了焦距和主点所在的坐标系统。在这里，默认值为"ndc_norm_image_bounds"，
                     表示焦距和主点在被归一化设备坐标（Normalized Device Coordinates, NDC）系统里。
 """
@dataclass
class ViewpointAnnotation:
    R: Tuple[TF3, TF3, TF3]  # 右乘 (PyTorch3D) 格式。 X_cam = X_world @ R + T
    T: TF3  # 平移向量
    focal_length: Tuple[float, float]  # 焦距
    principal_point: Tuple[float, float]  # 主点
    intrinsics_format: str = "ndc_norm_image_bounds"  # 定义焦距和主点所在的坐标系统

# 定义帧注解的数据类
@dataclass
class FrameAnnotation:
    """用于从json加载注解的数据类。"""
    sequence_name: str  # 用于连接 `SequenceAnnotation`
    frame_number: int  # 序列中的0基连续帧号
    frame_timestamp: float  # 从视频开始的时间戳（秒）
    image: ImageAnnotation  # 图像注解
    depth: Optional[DepthAnnotation] = None  # 深度注解
    mask: Optional[MaskAnnotation] = None  # 遮罩注解
    viewpoint: Optional[ViewpointAnnotation] = None  # 视点注解
    meta: Optional[Dict[str, Any]] = None  # 其他元数据

# 定义点云注解的数据类
@dataclass
class PointCloudAnnotation:
    path: str  # 相对于数据集根目录的ply文件路径，只包含点
    quality_score: float  # 质量评分，越高越好
    n_points: Optional[int]  # 点的数量

# 定义视频注解的数据类
@dataclass
class VideoAnnotation:
    path: str  # 相对于数据集根目录的原始视频文件路径
    length: float  # 视频长度（秒）

# 定义序列注解的数据类
@dataclass
class SequenceAnnotation:
    sequence_name: str  # 序列名称
    category: str  # 类别
    video: Optional[VideoAnnotation] = None  # 视频注解
    point_cloud: Optional[PointCloudAnnotation] = None  # 点云注解
    viewpoint_quality_score: Optional[float] = None  # 视点质量评分，越高越好

# 将数据类对象转储为json
def dump_dataclass(obj: Any, f: IO, binary: bool = False) -> None:
    """
    参数:
        f: 文件路径或已打开的文件对象。
        obj: 一个 @dataclass 或包含数据类的集合层次结构。
        binary: 如果 `f` 是文件句柄则设置为 True，否则为 False。
    """
    if binary:
        f.write(json.dumps(_asdict_rec(obj)).encode("utf8"))
    else:
        json.dump(_asdict_rec(obj), f)

# 从json加载到数据类对象
def load_dataclass(f: IO, cls: Type[_X], binary: bool = False) -> _X:
    """
    从json递归加载到 @dataclass 或包含数据类的集合层次结构。
    可以像 `load_dataclass(f, typing.List[FrameAnnotationAnnotation])` 一样调用它。
    如果json包含与数据类字段不对应的键则引发 KeyError。

    参数:
        f: 文件路径或已打开的文件对象。
        cls: 加载的数据类的类。
        binary: 如果 `f` 是文件句柄则设置为 True，否则为 False。
    """
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)

    if isinstance(asdict, list):
        # 在列表情况下运行更快的“向量化”版本
        cls = get_args(cls)[0]
        res = list(_dataclass_list_from_dict_list(asdict, cls))
    else:
        res = _dataclass_from_dict(asdict, cls)

    return res

# 向量化版本的 `_dataclass_from_dict`
 """
 这里的向量化是指这个函数处理的是一个"列表"的数据，通过这个函数，可以同一时间将多个数据（每个数据都是字典格式）反序列化成对应的数据类对象，
 而不是只处理一个数据。
  """

def _dataclass_list_from_dict_list(dlist, typeannot):
    """
    `_dataclass_from_dict` 的向量化版本。
    输出应等同于 `[_dataclass_from_dict(d, typeannot) for d in dlist]`。

    参数:
        dlist: 要转换的对象列表。
        typeannot: 这些对象的类型。
    返回:
        迭代器或转换对象的列表，与 `dlist` 长度相同。

    引发:
        ValueError: 假设对象在一致的位置有 None，否则会忽略一些值。
        这通常适用于自动生成的注解，但否则使用 `_dataclass_from_dict`。
    """
    cls = get_origin(typeannot) or typeannot

if typeannot is Any: # 如果目标类型是任何类型，那么直接返回
    return dlist
if all(obj is None for obj in dlist):  # 如果列表里的所有元素都是 None，同样直接返回
    return dlist
if any(obj is None for obj in dlist): # 如果列表里存在 None 元素
    # 过滤掉 None 元素并递归调用自身来处理剩余的元素
    idx_notnone = [(i, obj) for i, obj in enumerate(dlist) if obj is not None]
    idx, notnone = zip(*idx_notnone)
    converted = _dataclass_list_from_dict_list(notnone, typeannot)
    res = [None] * len(dlist)
    for i, obj in zip(idx, converted):
        res[i] = obj
    return res

is_optional, contained_type = _resolve_optional(typeannot) # 解析 typeannot 是否可选类型
if is_optional:
    return _dataclass_list_from_dict_list(dlist, contained_type) # 如果是，则处理具体的类型

# 根据提供的 typeannot 的实际类型，进行对应的转换
if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
    # 如果 cls 是一个 namedtuple，则会进入这个分支
    types = cls._field_types.values()
    dlist_T = zip(*dlist)
    res_T = [
        _dataclass_list_from_dict_list(key_list, tp)
        for key_list, tp in zip(dlist_T, types)
    ]
    return [cls(*converted_as_tuple) for converted_as_tuple in zip(*res_T)]
elif issubclass(cls, (list, tuple)): # 如果 cls 是列表或者元组
    types = get_args(typeannot)
    if len(types) == 1:  # 可能是 List；为所有项复制
        types = types * len(dlist[0])
    dlist_T = zip(*dlist)
    res_T = (_dataclass_list_from_dict_list(pos_list, tp) for pos_list, tp in zip(dlist_T, types))
    if issubclass(cls, tuple):
        return list(zip(*res_T))
    else:
        return [cls(converted_as_tuple) for converted_as_tuple in zip(*res_T)]
elif issubclass(cls, dict): # 如果 cls 是字典
    key_t, val_t = get_args(typeannot)
    all_keys_res = _dataclass_list_from_dict_list([k for obj in dlist for k in obj.keys()], key_t)
    all_vals_res = _dataclass_list_from_dict_list([v for obj in dlist for v in obj.values()], val_t)
    indices = np.cumsum([len(obj) for obj in dlist])
    assert indices[-1] == len(all_keys_res)

    keys = np.split(list(all_keys_res), indices[:-1])
    all_vals_res_iter = iter(all_vals_res)
    return [cls(zip(k, all_vals_res_iter)) for k in keys]
elif not dataclasses.is_dataclass(typeannot):
    return dlist # 如果都不是，直接返回原列表

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {
        f.name: (_unwrap_type(f.type), _get_dataclass_field_default(f))
        for f in dataclasses.fields(typeannot)
    }

    key_lists = (
        _dataclass_list_from_dict_list([obj.get(k, default) for obj in dlist], type_)
        for k, (type_, default) in fieldtypes.items()
    )
    transposed = zip(*key_lists)
    return [cls(*vals_as_tuple) for vals_as_tuple in transposed]

# 从字典中生成数据类对象
def _dataclass_from_dict(d, typeannot):
    if d is None or typeannot is Any:
        return d
    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        return _dataclass_from_dict(d, contained_type)

    cls = get_origin(typeannot) or typeannot
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        types = cls._field_types.values()
        return cls(*[_dataclass_from_dict(v, tp) for v, tp in zip(d, types)])
    elif issubclass(cls, (list, tuple)):
        types = get_args(typeannot)
        if len(types) == 1:  # 可能是 List；为所有项复制
            types = types * len(d)
        return cls(_dataclass_from_dict(v, tp) for v, tp in zip(d, types))
    elif issubclass(cls, dict):
        key_t, val_t = get_args(typeannot)
        return cls(
            (_dataclass_from_dict(k, key_t), _dataclass_from_dict(v, val_t))
            for k, v in d.items()
        )
    elif not dataclasses.is_dataclass(typeannot):
        return d

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {f.name: _unwrap_type(f.type) for f in dataclasses.fields(typeannot)}
    return cls(**{k: _dataclass_from_dict(v, fieldtypes[k]) for k, v in d.items()})

# 解除类型包装
def _unwrap_type(tp):
    if get_origin(tp) is Union:
        args = get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):  # noqa: E721
            return args[0] if args[1] is type(None) else args[1]  # noqa: E721
    return tp

# 获取数据类字段的默认值
def _get_dataclass_field_default(field: Field) -> Any:
    if field.default_factory is not MISSING:
        return field.default_factory()
    elif field.default is not MISSING:
        return field.default
    else:
        return None

# 递归地将数据类对象转换为字典
def _asdict_rec(obj):
    return dataclasses._asdict_inner(obj, dict)

# 将数据类对象转储为压缩的json文件
def dump_dataclass_jgzip(outfile: str, obj: Any) -> None:
    """
    将对象转储到gzipped json文件中。

    参数:
        obj: 一个 @dataclass 或包含数据类的集合层次结构。
        outfile: 输出文件的路径。
    """
    with gzip.GzipFile(outfile, "wb") as f:
        dump_dataclass(obj, cast(IO, f), binary=True)

# 从压缩的json文件中加载数据类对象
def load_dataclass_jgzip(outfile, cls):
    """
    从gzipped json文件中加载数据类。

    参数:
        outfile: 加载文件的路径。
        cls: 加载的数据类的类型注解。

    返回:
        加载的数据类对象。
    """
    with gzip.GzipFile(outfile, "rb") as f:
        return load_dataclass(cast(IO, f), cls, binary=True)

# 解析可选类型
def _resolve_optional(type_: Any) -> Tuple[bool, Any]:
    """检查 `type_` 是否等价于 `typing.Optional[T]`。"""
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_
