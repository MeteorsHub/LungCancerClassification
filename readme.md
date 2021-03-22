# 基于mRNA标志物表达量的肺癌亚型分类

## 运行环境

python3环境，可以根据运行时import包失败说明，使用pip或conda安装相应的依赖包。

## 目录结构

### 主要

* `/data`：数据，csv格式
* `/config`：yaml格式的实验设置，一个实验对应一个config，以不同模型划分成多个文件夹
* `/exp`：实验结果目录，根据yaml设置名称自动生成相同层级结构
* `data_loader.py`：数据读取，整理、预处理等，包括特征选择和特征变换的流程
* `train_eval.py`：实验主文件，包含训练模型和基准测试，结果存储到/exp目录中

### 次要

* `utils.py`：一些工具函数，例如初始化模型等
* `visualization.py`：一些绘图工具，主要用于生成基准测试结果图
* `feature_analysis.py`：使用1nn对原始特征进行可分性分析
* `run_all_train_eval.py`：批量运行多个实验
* `export_results.py`：拷贝实验结果到其他地方

## 运行示例

使用pycharm软件打开本工作目录可以自动加载一些运行配置。 若使用命令行运行，可以参考以下示例：

```shell
# -c参数表示使用\config文件夹中的哪个实验设置
# -s参数表示
python train_eval.py \
-c voting/joint_prime_feature_biclass \
-s 2svm_class0_fs_metric
-o
-r
```

