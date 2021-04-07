# ikcest比赛代码说明
介绍ikcest比赛中用到的数据、代码、文件，如何配置环境以及如何复现结果。

# 基本信息
队伍名称：黄渡龙舟队
参赛队员: 农婧颖，崔啸萱，姚霖，张瑜
初赛排名：14/3024  最终排名：8/3024

## 1.文件树
[code_project](code_project)
- code_project
    - [code](code_project/code)存放存放比赛用到的源代码
    - [environment.yaml](code_project/environment.yaml)代码运行所需的环境依赖。
    - [说明文档](code_project/说明文档)存放模型需要的输入数据
- 说明文档: 对环境依赖、代码运行情况的说明
- generate.sh: 复现结果的主脚本文件

[code](code)
- code
    - [train_data](code/train_data)存放Curve_linear_fit.py拟合过程中用到的数据。
    - [models](code/models)存放模型的定义及实现py文件。
    - [data](code/data)存放模型需要的输入数据
        - `data`文件夹，存放模型需要的infection数据，30*903的矩阵形式。
        - `adj`文件夹，模型用到的邻接矩阵，30*903的矩阵形式。
        - `predict.py`文件，预测未来30天的代码。
        - `pre_utils.py`文件，预测未来30天时的数据获取代码。
    - [predict](code/predict)存放运行脚本时，模型的输出结果
        - `submission_final.csv`文件，最后复现出来的提交结果文件。
    - [mysave](code/mysave)存放预训练模型，模型参数文件，包括attention文件。
    - [adj_generate](code/adj_generate)存放邻接矩阵生成的代码，adj文件夹中的邻接矩阵由这生成。
        - `src4`文件夹，存放生成后的邻接矩阵文件。
        - `final_adjacent_matrix.py`文件，邻接矩阵生成代码，需要单独调用。
        - `adj_env.yaml`文件，邻接矩阵生成代码需要用到的环境，需要单独使用。

    - [generate.sh](code/generate.sh)主脚本，通过运行该脚本复现结果，全局入口。
    - [main.py](code/main.py)模型训练、预测的入口函数
    - [Optim.py](code/Optim.py)存放各个预训练模型，模型参数文件。
    - [utils.py](code/utils.py)存放各个预训练模型，模型参数文件。
    - [Curve_linear_V2.py](code/Curve_linear_V2.py)模型训练、预测的入口函数
    - [post_proc_single.py](code/post_proc_single.py)格式转换代码，主要是将模型预测原始结果转化为标准提交格式。
    - [post_proc_update.py](code/post_proc_update.py)不同模型结果融合代码。
    - [zone_all.csv](code/zone_all.csv)格式化转换是需要用到的文件。
    - [main1.py](code/main1.py)训练AR模型时用到的代码


## 2.环境配置
### 2.1硬件运行环境
CPU：Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz
GPU:  NVIDIA 1080Ti 11GB*4
RAM：32GB

### 2.2软件运行环境
操作系统：Ubuntu 18.04.3 LTS
编程语言：Python 2.7
环境依赖：

部分环境依赖如下：
python=2.7.18, pytorch=1.0.0,cudatoolkit=10.2.89,torchvision=0.2.2,sympy=1.5,pandas=0.24.2,scipy=1.2.1等等
**[推荐]** 我们制作了一个环境配置文件，可以基于anaconda，使用environment.yaml来快速创建复现实验结果所需环境，具体指令如下：
`conda env create -f environment.yaml`

如果创建上述环境过程中遇到了prefix相关的问题，请考虑修改environment.yaml文件中的name内容或者prefix路径，使之与您本地的路径保持一致。

## 3.结果复现
运行generate.sh脚本即可复现结果，具体指令如下：
`chmod +x ./generate.sh`
`./generate.sh`

其中最后的结果文件存放在./predict/submission_final.csv中
注意：如果创建上述环境过程中遇到了prefix相关的问题，请考虑修改environment.yaml文件中的name内容或者prefix路径，使之与您本地的路径保持一致。

## 4.注意事项
- 在复现实验结果的时候，使用的是已经生成好的邻接矩阵，只需要激活environment.yaml文件中的依赖即可。
- 如果需要单独生成邻接矩阵，由于使用到的依赖与environment.yaml环境中的依赖存在冲突，需要创建另外一个环境该环境的相关环境依赖如下：
`Python=3.6, pandas=1.0.3, numpy=1.18.2, scipy=1.4.1`

在新建了以上新的环境后，可以通过调用并运行`code/adj_generate`目录下的`final_adjacent_matrix.py`文件，得到邻接矩阵，对应的邻接矩阵文件保存在`code/adj_generate/src4`文件夹中


