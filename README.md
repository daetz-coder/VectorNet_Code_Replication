# VectorNet_Code_Replication



# 车辆轨迹预测系列 (四)：VectorNet代码复现及踩坑记录

车辆轨迹预测系列 (四)：VectorNet代码复现及踩坑记录

[车辆轨迹预测系列 (四)：VectorNet代码复现及踩坑记录]()[一、准备]()[1、Argoverse API 下载]()[1) clone]()[2） Download HD map data]()[3） pip argoverse-api]()[2、vectornet 下载]()[1） clone]()[2） download dataset]()[3）feature preprocessing]()[3、安装PyG]()[1）Additional Libraries]()[2）pip torch-geometric ]()[3）train]()[二、踩坑记录]()[1、Segmentation fault (core dumped)]()[2、Plain typing_extensions.Self is not valid as type argument]()[3、AssertionError: Invalid device id]()[4、&#39;data.DataListLoader&#39; is deprecated, use &#39;loader.DataListLoader&#39; instead]()[5、TypeError:  **inc** () takes 3 positional arguments but 4 were given]()[6、 Output 0 of SliceBackward0 is a view and is being modified inplace.]()[三、说明]()

**本文用于复现VectorNet中的具体实验，由于官方没有提供具体的代码，在本文中选择一个"民间版本"进行复现，**

**具体地址如下：**[yet-another-vectornet](https://github.com/xk-huang/yet-another-vectornet)

**在复现过程中尽量保证使用** `.ipynb`方便理解和展示输出的结果

**项目地址：**[GitHub - daetz-coder/VectorNet_Code_Replication](https://github.com/daetz-coder/VectorNet_Code_Replication)

## 一、准备

* [Github-argoverse数据集的api地址](https://github.com/argoverse/argoverse-api)
* [argoverse-forecasting-link argoverse预测数据集](https://www.argoverse.org/av1.html#forecasting-link)

**一般来说是需要下载完整的数据集，为了方便起见，我这里仅上传几个数据（mini版本），可根据需求下载完整数据集。**

![image-20240626105431003](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261054138.png)

### 1、Argoverse API 下载

**！！！注意：目前这个api仅支持Linux和MacOS，暂不支持Windows** ，笔者后续均使用AutoDL云平台演示

* [Github-argoverse数据集的api地址](https://github.com/argoverse/argoverse-api)

![image-20240626105731742](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261057807.png)

#### 1) clone

```
git clone https://github.com/argoai/argoverse-api.git
```

![image-20240626111205726](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261112785.png)

#### 2） Download HD map data

**Download **`hd_maps.tar.gz` from [our website](https://www.argoverse.org/av1.html#download-link) and extract into the root directory of the repo. Your directory structure should look something like this:

**需要去官网下载地图：[Miami and Pittsburgh](**[https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/hd_maps.tar.gz](https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/hd_maps.tar.gz)

![image-20240626111251716](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261112781.png)

**解压后最后的目录结构是**

```
argodataset
└── argoverse
    └── data_loading
    └── evaluation
    └── map_representation
    └── utils
    └── visualization
└── map_files
└── license
...
```

**选上传地图文件与argodataset-api同级，再使用下述命令解压**

```
tar -xzf hd_maps.tar.gz -C argoverse-api
```

`tar`：这是 Unix/Linux 系统中的一个命令，用于创建和处理归档文件（tarballs）。

`-x`：这是 `tar` 命令的一个选项，表示要解压归档文件。`x` 代表 extract（解压）。

`-z`：这是 `tar` 命令的另一个选项，表示归档文件使用了 gzip 压缩。`z` 代表 gzip。

`-f hd_maps.tar.gz`：这个选项告诉 `tar` 命令要操作的文件是 `hd_maps.tar.gz`。`f` 代表 file（文件），后面紧跟要操作的文件名。

`-C argoverse-api`：这是 `tar` 命令的选项，表示将解压后的文件提取到指定的目录 `argoverse-api` 中。`C` 代表 change directory（更改目录）。

![image-20240626112431237](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261124305.png)

#### 3） pip argoverse-api

**使用命令安装**

```
!pip install -e argoverse-api
```

```
Looking in indexes: http://mirrors.aliyun.com/pypi/simple
Obtaining file:///root/autodl-tmp/TrajectoryPrediction/VectorNet/argoverse-api
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
    Preparing wheel metadata ... done
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.13.1 requires numpy<=1.24.3,>=1.22, but you have numpy 1.19.0 which is incompatible.
seaborn 0.13.1 requires numpy!=1.24.0,>=1.20, but you have numpy 1.19.0 which is incompatible.
nuscenes-devkit 1.1.11 requires numpy>=1.22.0, but you have numpy 1.19.0 which is incompatible.
Successfully installed PyYAML-6.0.2rc1 antlr4-python3-runtime-4.8 argoverse-1.1.0 colour-0.1.5 hydra-core-1.1.0 imageio-2.34.2 lapsolver-1.1.0 llvmlite-0.39.1 motmetrics-1.1.3 numba-0.56.4 numpy-1.19.0 omegaconf-2.1.0 pandas-1.4.4 polars-0.20.31 pyntcloud-0.3.1 scipy-1.9.3 sklearn-0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

![image-20240626120515425](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261205491.png)

**尝试切换版本修复，升级将 **`numpy` 升级到 `1.22.0` 或更新版本。

```
pip install numpy==1.22.0
```

```
Successfully uninstalled numpy-1.19.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
argoverse 1.1.0 requires numpy==1.19, but you have numpy 1.22.0 which is incompatible.
Successfully installed numpy-1.22.0
```

![image-20240626122538851](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261225929.png)

**修改版本后，发现这里面的** `setup.py`指定了版本是 `1.19`导致安装冲突，替换成自己的版本即可 `1.22`

![image-20240626122606099](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261226167.png)

### 2、vectornet 下载

#### 1） clone

```
git clone https://github.com/xk-huang/yet-another-vectornet.git
```

![image-20240626112856856](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261128920.png)

#### 2） download dataset

**下载数据集，由于数据集较大，为了方便我仅仅选择了一小部分数据，具体内容如下**

![image-20240626124059408](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261240471.png)

![image-20240626124411948](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261244022.png)

**建议使用** `data`作为命名数据集文件，否则需要修改对应文件

#### 3）feature preprocessing

```
!python compute_feature_module.py
```

![image-20240626124826199](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261248277.png)

**发现仅仅处理了val，检查后发现需要注释** `compute_feature_module.py`的一段内容

![image-20240626125729046](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261257123.png)

**删除之前生成的内容,重新运行之前的代码**

```
!rm -rf ./interm_data/
```

![image-20240626125956893](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261259974.png)

### 3、安装PyG

#### 1）Additional Libraries

**注意这部分别使用pip直接安装**

**官方文档**[https://github.com/pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)

**首先检查直接的torch版本**

```
import torch

# Check the PyTorch version
print("PyTorch version:", torch.__version__)
```

```
PyTorch version: 1.10.0+cu113
```

![image-20240626154550822](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261545908.png)

![image-20240626152542565](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261525651.png)

**检查python版本**

```
import sys

# 获取并打印Python的主要版本号、次要版本号和微小版本号
version_info = sys.version_info
print(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
```

```
Python version: 3.8.10
```

**千万别选错了，笔者这里面下错了，应该是1.10的版本,第一次下成了1.11的版本，下错了就会报** `Segmentation fault (core dumped)`

![image-20240626171644089](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261716180.png)

**这里面每一个类型的都需要下载一个，如下所示**

![image-20240626184947064](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261849125.png)

**使用** `pip`导入,使用 `*`通配符全部加载

```
pip install ./whl/*.whl
```

![image-20240626153920379](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261539444.png)

#### 2）pip torch-geometric

**From ****PyG 2.3** onwards, you can install and use PyG **without any external library** required except for PyTorch. For this, simply run

**从PyG 2.3开始，您可以安装和使用PyG，而不需要任何外部库(PyTorch除外)。 为此，只需运行即可**

```
!pip install torch-geometric
```

**注意这里需要看你的版本，使用这行命令会下载最新版本，笔者亲测PyTorch version: 1.10.0+cu113+Python version: 3.8.10，使用最高版本（目前是2.5.3）会报 `Plain typing_extensions.Self is not valid as type argument`**

**使用下述命令可解决，具体内容见** `2、Plain typing_extensions.Self is not valid as type argument`

```
!pip install torch-geometric==2.0.1
```

#### 3）train

**这部分的问题特别的，具体内容需要结合** `二、踩坑记录`观看

```
!python train.py
```

**1、IndexError: list index out of range**

![image-20240626193834553](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261938625.png)

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead
  warnings.warn(out)
Traceback (most recent call last):
  File "train.py", line 92, in <module>
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/nn/data_parallel.py", line 43, in __init__
    self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))
IndexError: list index out of range
```

**关于 **`IndexError`这个问题发生在尝试指定 GPU 设备时。首先，确保你的系统中有可用的 CUDA 设备，并且你正确地指定了设备编号。你可以使用以下代码来检查系统中的 CUDA 设备数量和状态：

**方案A: 多上几个GPU**

**方案B：使用** `single_one.py`这里提供单卡运行，笔者这里改成使用 `single_one.py`

**2、显示AssertionError: Invalid device id**

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead
  warnings.warn(out)
Traceback (most recent call last):
  File "train.py", line 92, in <module>
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/nn/data_parallel.py", line 42, in __init__
    super(DataParallel, self).__init__(module, device_ids, output_device)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/parallel/data_parallel.py", line 142, in __init__
    _check_balance(self.device_ids)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/parallel/data_parallel.py", line 23, in _check_balance
    dev_props = _get_devices_properties(device_ids)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/_utils.py", line 464, in _get_devices_properties
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/root/miniconda3/lib/python3.8/site-packages/torch/_utils.py", line 464, in <listcomp>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/root/miniconda3/lib/python3.8/site-packages/torch/_utils.py", line 447, in _get_device_attr
    return get_member(torch.cuda)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/_utils.py", line 464, in <lambda>
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]
  File "/root/miniconda3/lib/python3.8/site-packages/torch/cuda/__init__.py", line 359, in get_device_properties
    raise AssertionError("Invalid device id")
AssertionError: Invalid device id
```

**具体原因：源代码使用系统中的第三和第四个 GPU（索引从0开始）。如果这些 GPU 设备不存在，就会出现这个错误。还是和之前一样，运行**

`single_one.py`

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead
  warnings.warn(out)
Traceback (most recent call last):
  File "train.py", line 111, in <module>
    out = model(data)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/nn/data_parallel.py", line 55, in forward
    data = Batch.from_data_list(
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/data/batch.py", line 63, in from_data_list
    batch, slice_dict, inc_dict = collate(
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py", line 76, in collate
    value, slices, incs = _collate(attr, values, data_list, stores,
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py", line 142, in _collate
    incs = get_incs(key, values, data_list, stores)
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py", line 187, in get_incs
    repeats = [
  File "/root/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py", line 188, in <listcomp>
    data.__inc__(key, value, store)
TypeError: __inc__() takes 3 positional arguments but 4 were given
```

**3、UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead**

**先解决警告的问题,警告信息表明 **`DataListLoader` 类的导入路径已经被弃用，建议使用新的导入路径。这是因为在 `torch_geometric` 的最新版本中，部分组件的结构可能已经被重新组织以优化库的结构和使用。为了修复这个警告并确保你的代码与最新的库版本兼容，你需要更新 `DataListLoader` 的导入语句。

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead
  warnings.warn(out)
```

![image-20240626200515883](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262005942.png)

**4、TypeError: **inc**() takes 3 positional arguments but 4 were given**

```
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_1034/1414215582.py in <module>
    104         num_samples = 1
    105         start_tic = time.time()
--> 106         for data in train_loader:
    107             if epoch < end_epoch: break
    108             if isinstance(data, List):

~/miniconda3/lib/python3.8/site-packages/torch/utils/data/dataloader.py in __next__(self)
    519             if self._sampler_iter is None:
    520                 self._reset()
--> 521             data = self._next_data()
    522             self._num_yielded += 1
    523             if self._dataset_kind == _DatasetKind.Iterable and \

~/miniconda3/lib/python3.8/site-packages/torch/utils/data/dataloader.py in _next_data(self)
    559     def _next_data(self):
    560         index = self._next_index()  # may raise StopIteration
--> 561         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    562         if self._pin_memory:
    563             data = _utils.pin_memory.pin_memory(data)

~/miniconda3/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50         else:
     51             data = self.dataset[possibly_batched_index]
---> 52         return self.collate_fn(data)

~/miniconda3/lib/python3.8/site-packages/torch_geometric/loader/dataloader.py in __call__(self, batch)
     37 
     38     def __call__(self, batch):
---> 39         return self.collate(batch)
     40 
     41 

~/miniconda3/lib/python3.8/site-packages/torch_geometric/loader/dataloader.py in collate(self, batch)
     17         elem = batch[0]
     18         if isinstance(elem, Data) or isinstance(elem, HeteroData):
---> 19             return Batch.from_data_list(batch, self.follow_batch,
     20                                         self.exclude_keys)
     21         elif isinstance(elem, torch.Tensor):

~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/batch.py in from_data_list(cls, data_list, follow_batch, exclude_keys)
     61         Will exclude any keys given in :obj:`exclude_keys`."""
     62 
---> 63         batch, slice_dict, inc_dict = collate(
     64             cls,
     65             data_list=data_list,

~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py in collate(cls, data_list, increment, add_batch, follow_batch, exclude_keys)
     74 
     75             # Collate attributes into a unified representation:
---> 76             value, slices, incs = _collate(attr, values, data_list, stores,
     77                                            increment)
     78 

~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py in _collate(key, values, data_list, stores, increment)
    140         slices = cumsum([value.size(cat_dim or 0) for value in values])
    141         if increment:
--> 142             incs = get_incs(key, values, data_list, stores)
    143             if incs.dim() > 1 or int(incs[-1]) != 0:
    144                 values = [value + inc for value, inc in zip(values, incs)]

~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py in get_incs(key, values, data_list, stores)
    185 def get_incs(key, values: List[Any], data_list: List[BaseData],
    186              stores: List[BaseStorage]) -> Tensor:
--> 187     repeats = [
    188         data.__inc__(key, value, store)
    189         for value, data, store in zip(values, data_list, stores)

~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py in <listcomp>(.0)
    186              stores: List[BaseStorage]) -> Tensor:
    187     repeats = [
--> 188         data.__inc__(key, value, store)
    189         for value, data, store in zip(values, data_list, stores)
    190     ]

TypeError: __inc__() takes 3 positional arguments but 4 were given
```

* [https://github.com/pyg-team/pytorch_geometric/issues/6779](https://github.com/pyg-team/pytorch_geometric/issues/6779)

**使用** `def __inc__(self, key, value, *args, **kwargs):`替换原始的 `def __inc__(self, key, value):`即可

![image-20240626212448044](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262124125.png)

**5、RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace.**

```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_1228/1414215582.py in <module>
    112                 y = data.y.view(-1, out_channels)
    113             optimizer.zero_grad()
--> 114             out = model(data)
    115             loss = F.mse_loss(out, y)
    116             loss.backward()

~/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/vectornet.py in forward(self, data)
     54         sub_graph_out = self.subgraph(data)
     55         x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
---> 56         out = self.self_atten_layer(x, valid_lens)
     57         # from pdb import set_trace
     58         # set_trace()

~/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/selfatten.py in forward(self, x, valid_len)
     57         value = self.v_lin(x)
     58         scores = torch.bmm(query, key.transpose(1, 2))
---> 59         attention_weights = masked_softmax(scores, valid_len)
     60         return torch.bmm(attention_weights, value)

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/selfatten.py in masked_softmax(X, valid_len)
     32         X = X.reshape(-1, shape[-1])
     33         for count, row in enumerate(X):
---> 34             row[int(valid_len[count]):] = -1e6
     35         return nn.functional.softmax(X.reshape(shape), dim=-1)
     36 

RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
```

**这个错误 **`RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace.` 是由于在 PyTorch 中直接修改视图（view）引起的，尤其是当视图作为某个函数（返回多个视图的函数）的输出时。在这种情况下，PyTorch 需要保证梯度计算的正确性，因此不允许直接就地修改视图。

```
# def masked_softmax(X, valid_len):
#     """
#     masked softmax for attention scores
#     args:
#         X: 3-D tensor, valid_len: 1-D or 2-D tensor
#     """
#     if valid_len is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_len.dim() == 1:
#             valid_len = torch.repeat_interleave(
#                 valid_len, repeats=shape[1], dim=0)
#         else:
#             valid_len = valid_len.reshape(-1)
#         # Fill masked elements with a large negative, whose exp is 0
#         X = X.reshape(-1, shape[-1])
#         for count, row in enumerate(X):
#             row[int(valid_len[count]):] = -1e6
#         return nn.functional.softmax(X.reshape(shape), dim=-1)
def masked_softmax(X, valid_len):
    """
    Masked softmax for attention scores.
    Args:
        X: 3-D tensor of shape (batch_size, num_heads, sequence_length)
        valid_len: 1-D or 2-D tensor with the valid lengths of each sequence.
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = valid_len.unsqueeze(1).expand(-1, shape[2])
        else:
            valid_len = valid_len.reshape(-1)

        # Create a mask based on the sequences' valid lengths
        mask = torch.arange(shape[-1], device=X.device)[None, :] < valid_len[:, None]

        # Use the mask to modify the elements of X before the softmax
        X = X.clone().masked_fill(~mask, -1e9)  # Use a large negative value that approximates to zero in softmax

        return nn.functional.softmax(X, dim=-1)
```

1. **扩展 valid_len**：如果 `valid_len` 是一维的，则通过 `unsqueeze` 和 `expand` 扩展到与 `X` 的最后一个维度匹配。
2. **创建布尔掩码**：使用 `torch.arange` 和比较操作创建一个布尔掩码，这个掩码直接表明每个元素是否应该被包括在 softmax 计算中。
3. **使用掩码修改数据**：利用 `masked_fill` 方法和掩码来将 `X` 中应被忽略的元素设置为一个大的负值（`-1e9`），该操作不是就地操作，避免了修改视图所带来的问题。
4. **执行 softmax 操作**：最后，将修改后的 `X` 传递给 `softmax` 函数进行归一化计算。

**这样的处理不仅解决了潜在的就地修改问题，还提高了函数的效率和可读性。此外，使用 **`masked_fill` 是处理这类问题的推荐方法，因为它在底层优化了性能，避免了不必要的 Python 层面的循环。

**修复之后成功运行**

![image-20240626214045771](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262140841.png)

## 二、踩坑记录

### 1、Segmentation fault (core dumped)

**现象：在** `import torch_geometric`直接报错 `Segmentation fault (core dumped)`

**解释：千万别选错了，笔者这里面下错了，应该是1.10的版本第一次下成了1.11的版本去了，下错了就会报** `Segmentation fault (core dumped)`

**参考：**[https://github.com/pyg-team/pytorch_geometric/issues/4363](https://github.com/pyg-team/pytorch_geometric/issues/4363)

![image-20240626171644089](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261716180.png)

**安装完成之后使用继续运行代码**

```
!python train.py
```

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/typing.py:54: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: /root/miniconda3/lib/python3.8/site-packages/libpyg.so: undefined symbol: _ZNK3c104Type14isSubtypeOfExtERKS0_PSo
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
```

![image-20240626171222456](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261712510.png)

**出现错误，检查发现和**[https://github.com/pyg-team/pytorch_geometric/issues/4363](https://github.com/pyg-team/pytorch_geometric/issues/4363)非常相似

### 2、Plain typing_extensions.Self is not valid as type argument

![image-20240626185345251](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261853304.png)

**怀疑是** `torch_geometric` 可能与你的 Python 版本不完全兼容。通常，最新的 `torch_geometric` 版本支持最新的 Python 版本。请检查你的 Python 版本与 `torch_geometric` 的兼容性。

```
!pip install torch-geometric==2.0.1
```

![image-20240626185650490](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261856558.png)

![image-20240626185700669](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406261857847.png)

**有效，成功解决!**

### 3、AssertionError: Invalid device id

**具体原因：源代码使用系统中的第三和第四个 GPU（索引从0开始）。如果这些 GPU 设备不存在，就会出现这个错误。**

**可以使用** `single_one.py`这里提供单卡/无卡运行，笔者这里改成使用 `single_one.py`即可

### 4、'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead

**警告信息表明 **`DataListLoader` 类的导入路径已经被弃用，建议使用新的导入路径。这是因为在 `torch_geometric` 的最新版本中，部分组件的结构可能已经被重新组织以优化库的结构和使用。为了修复这个警告并确保你的代码与最新的库版本兼容，你需要更新 `DataListLoader` 的导入语句。

```
/root/miniconda3/lib/python3.8/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataListLoader' is deprecated, use 'loader.DataListLoader' instead
  warnings.warn(out)
```

![image-20240626200515883](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262005942.png)

### 5、TypeError: **inc**() takes 3 positional arguments but 4 were given

```
~/miniconda3/lib/python3.8/site-packages/torch_geometric/data/collate.py in <listcomp>(.0)
    186              stores: List[BaseStorage]) -> Tensor:
    187     repeats = [
--> 188         data.__inc__(key, value, store)
    189         for value, data, store in zip(values, data_list, stores)
    190     ]

TypeError: __inc__() takes 3 positional arguments but 4 were given
```

[https://github.com/pyg-team/pytorch_geometric/issues/6779](https://github.com/pyg-team/pytorch_geometric/issues/6779)

**具体内容如下，由于之前安装的版本是2.0.1大于2.0，导致这部分的代码需要进行修改**

```
def __inc__(self, key, value, *args, **kwargs):
```

![image-20240626212138105](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262121220.png)

![](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262124125.png)

### 6、 Output 0 of SliceBackward0 is a view and is being modified inplace.

```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_1228/1414215582.py in <module>
    112                 y = data.y.view(-1, out_channels)
    113             optimizer.zero_grad()
--> 114             out = model(data)
    115             loss = F.mse_loss(out, y)
    116             loss.backward()

~/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/vectornet.py in forward(self, data)
     54         sub_graph_out = self.subgraph(data)
     55         x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
---> 56         out = self.self_atten_layer(x, valid_lens)
     57         # from pdb import set_trace
     58         # set_trace()

~/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1101                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1102             return forward_call(*input, **kwargs)
   1103         # Do not call functions when jit is used
   1104         full_backward_hooks, non_full_backward_hooks = [], []

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/selfatten.py in forward(self, x, valid_len)
     57         value = self.v_lin(x)
     58         scores = torch.bmm(query, key.transpose(1, 2))
---> 59         attention_weights = masked_softmax(scores, valid_len)
     60         return torch.bmm(attention_weights, value)

~/autodl-tmp/TrajectoryPrediction/VectorNet/yet-another-vectornet/modeling/selfatten.py in masked_softmax(X, valid_len)
     32         X = X.reshape(-1, shape[-1])
     33         for count, row in enumerate(X):
---> 34             row[int(valid_len[count]):] = -1e6
     35         return nn.functional.softmax(X.reshape(shape), dim=-1)
     36 

RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
```

**这个错误 **`RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace.` 是由于在 PyTorch 中直接修改视图（view）引起的，尤其是当视图作为某个函数（返回多个视图的函数）的输出时。在这种情况下，PyTorch 需要保证梯度计算的正确性，因此不允许直接就地修改视图。

```
# def masked_softmax(X, valid_len):
#     """
#     masked softmax for attention scores
#     args:
#         X: 3-D tensor, valid_len: 1-D or 2-D tensor
#     """
#     if valid_len is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_len.dim() == 1:
#             valid_len = torch.repeat_interleave(
#                 valid_len, repeats=shape[1], dim=0)
#         else:
#             valid_len = valid_len.reshape(-1)
#         # Fill masked elements with a large negative, whose exp is 0
#         X = X.reshape(-1, shape[-1])
#         for count, row in enumerate(X):
#             row[int(valid_len[count]):] = -1e6
#         return nn.functional.softmax(X.reshape(shape), dim=-1)
def masked_softmax(X, valid_len):
    """
    Masked softmax for attention scores.
    Args:
        X: 3-D tensor of shape (batch_size, num_heads, sequence_length)
        valid_len: 1-D or 2-D tensor with the valid lengths of each sequence.
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = valid_len.unsqueeze(1).expand(-1, shape[2])
        else:
            valid_len = valid_len.reshape(-1)

        # Create a mask based on the sequences' valid lengths
        mask = torch.arange(shape[-1], device=X.device)[None, :] < valid_len[:, None]

        # Use the mask to modify the elements of X before the softmax
        X = X.clone().masked_fill(~mask, -1e9)  # Use a large negative value that approximates to zero in softmax

        return nn.functional.softmax(X, dim=-1)
```

1. **扩展 valid_len**：如果 `valid_len` 是一维的，则通过 `unsqueeze` 和 `expand` 扩展到与 `X` 的最后一个维度匹配。
2. **创建布尔掩码**：使用 `torch.arange` 和比较操作创建一个布尔掩码，这个掩码直接表明每个元素是否应该被包括在 softmax 计算中。
3. **使用掩码修改数据**：利用 `masked_fill` 方法和掩码来将 `X` 中应被忽略的元素设置为一个大的负值（`-1e9`），该操作不是就地操作，避免了修改视图所带来的问题。
4. **执行 softmax 操作**：最后，将修改后的 `X` 传递给 `softmax` 函数进行归一化计算。

**、**![image-20240626212600250](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202406262126365.png)

## 三、说明

* **为了方便存放，本文使用的数据量较少，用户可自行补充数据到data文件夹下**
* **由于没有官方源码，并且相关的依赖文件历经多次更新，代码中可能还存在部分问题**
* **后续会具体对文章和代码进行解释说明**
* **所有文件包括数据集，均已存入git中，包括相关的** `.ipynb`用于展示具体的过程
* **相关代码已经上传（已提供少量data数据）**[GitHub - daetz-coder/VectorNet_Code_Replication: VectorNet Code Replication，Contains mini data datasets that can be run directly，The visualization content will be updated in the future](https://github.com/daetz-coder/VectorNet_Code_Replication)

**参考：**

* [不愧是公认最好的【图神经网络GNN/GCN教程】，从基础到进阶再到实战，一个合集全部到位！-人工智能/神经网络/图神经网络/深度学习。*哔哩哔哩*bilibili](https://www.bilibili.com/video/BV1184y1x71H/?spm_id_from=333.337.search-card.all.click)
* [【计算机视觉】VectorNet论文解读，基于图神经网络的驾驶轨迹预测，简直不要太强！！无人驾驶|深度学习|人工智能*哔哩哔哩*bilibili](https://www.bilibili.com/video/BV1XV4y1M7Y4/?spm_id_from=333.337.search-card.all.click&vd_source=591db91b15fb99b2280157ce6d306c42)
* [VectorNet - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/681237627)
* [GitHub - xk-huang/yet-another-vectornet: Vectornet for trajectory prediction, implemented in PyTorch/Torch_geometric](https://github.com/xk-huang/yet-another-vectornet)
* **[**[2005.04259]** VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation (arxiv.org)**](https://arxiv.org/abs/2005.04259)
* [GitHub - argoverse/argoverse-api: Official GitHub repository for Argoverse dataset](https://github.com/argoverse/argoverse-api)
* [Home (argoverse.org)](https://www.argoverse.org/)
* [GitHub - pyg-team/pytorch_geometric: Graph Neural Network Library for PyTorch](https://github.com/pyg-team/pytorch_geometric/)
* [PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/)
* [Segmentation fault (core dumped) on import torch_geometric · Issue #4363 · pyg-team/pytorch_geometric · GitHub](https://github.com/pyg-team/pytorch_geometric/issues/4363)
* [DataLoader Error · Issue #6779 · pyg-team/pytorch_geometric · GitHub](https://github.com/pyg-team/pytorch_geometric/issues/6779)
