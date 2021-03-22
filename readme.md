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
# -c参数表示使用/config文件夹中的哪个实验设置
# -s参数为当前config文件下的一个子实验标题，可以任取
# -o参数表示使用/config文件夹中的设置文件，删除该参数则使用/exp文件夹中对应实验存储的设置文件
# -r参数表示重新训练模型，删除该参数可以在/exp文件夹中已有模型存储时报错，防止误删模型
python train_eval.py \
-c voting/joint_prime_feature_biclass \
-s 2svm_class0_fs_metric \
-o \
-r
```

上述示例中，`-c`参数使用的config命名规则是： voting表示模型名称，joint表示同时使用所有标志物特征， prime_feature表示只使用主要特征（ave_fold和sd）， biclass表示这是一个二分类任务

代码运行完成后，将在/exp文件夹中的对应位置存储一些实验结果，其目录结构为：

* `/model`：存储训练好的模型
* `best_params.yaml`：超参搜索得到的最佳超参
* `conf_mat.png`：有关该实验的一些结果图，例如特征可视化、roc曲线等
* `config.yaml`：备份此次实验的设置
* `feature_selection_and_transformation.txt`：存储特征选择和特征变换的一些结果
* `metrics.txt`：整体的实验指标

## 最佳实验设置

目前的v1版本里，二分类实验的最佳结果所用的实验设置放在/config/archive/v1里，使用了特征选择、特征转换和svm分类器。 括号里为训练和测试auc

0. 正常（87.4，81.1）
    ```shell
    python train_eval.py \
    -c archive/v_1/biclass_0 \
    -s fs_ft_svm
    ```
1. 腺癌（86.2，80.7）
    ```shell
    python train_eval.py \
    -c archive/v_1/biclass_1 \
    -s fs_ft_svm
    ```
2. 鳞癌（86.9，83.8）
    ```shell
    python train_eval.py \
    -c archive/v_1/biclass_2 \
    -s fs_ft_svm
    ```
3. 小细胞癌（99.7，98.9）
    ```shell
    python train_eval.py \
    -c archive/v_1/biclass_3 \
    -s fs_ft_svm
    ```
5. 转移癌（97.4，93.2）
    ```shell
    python train_eval.py \
    -c archive/v_1/biclass_5 \
    -s fs_ft_svm
    ```