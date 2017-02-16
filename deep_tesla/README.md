# Deep Learning for Self-Driving Cars : DeepTesla
http://selfdrivingcars.mit.edu/deeptesla/

## 题目描述
本项目中，你需要利用MIT 6.S094 这门公开课中的Tesla数据集训练深度学习模型，根据车辆的前置相机所拍摄的路况图像，实现对车辆转向角度的预测。

![](./img/gif_tesla.gif)

## 数据
主要包括tesla在两种不同驾驶模式（human driving和autopilot）下的前置相机录制的视频和车辆的转向控制信号。数据可以从这里下载:[百度云](https://pan.baidu.com/s/1hrEWtyG)

数据格式如下:

  - 前置相机记录的视频: 截图如下
      ![](./img/video_vis1.png)
  
  - 行驶过程中的控制信号: CSV格式
  
  
       ts_micro         | frame_index | wheel 
      ------------------|-------------|-------
       1464305394391807 | 0           | -0.5  
       1464305394425141 | 1           | -0.5  
       1464305394458474 | 2           | -0.5  
      
    
    其中，`ts_micro`是时间戳，`frame_index`是帧编号，`wheel`是转向角度（以水平方向为基准，+为顺时针，-为逆时针）


## 建议
* [课程讲义](https://www.dropbox.com/s/q34bi7t0udms01x/lecture3.pdf?dl=1)提供了很好的入门介绍，原链接在dropbox，国内用户可以从[百度云](https://pan.baidu.com/s/1i472Jad)下载。
* [课程项目介绍](http://selfdrivingcars.mit.edu/deeptesla/)阐述了实现思路(ConvNetJS)。
* [课程网页应用](http://selfdrivingcars.mit.edu/deepteslajs/)提供了试验环境，可以测试模型的效果。
* 除了课程介绍之外的几个有用的课程和论文:
    - [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    - [Dataset and code for 2016 paper "Learning a Driving Simulator" ](https://github.com/commaai/research/blob/master/train_steering_model.py)
    - [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-1/)

* 注：作为Udacity MLND的毕业项目，我们建议使用keras+tensorflow+jupyter notebook来完成，但也可以同时利用ConvNetJS来进行测试(关于ConvNetJS的资料，可以参考[karpathy/convnetjs/](http://cs.stanford.edu/people/karpathy/convnetjs/))。

## 要求
* 优化你的模型并使预测结果和真实的转向角度尽可能接近。

* 这里我们以`epoch10_front.mkv`和`epoch10_steering.csv`对应的行驶记录作为评价标准，你的报告中应该包含模型最后在该记录中取得的效果。

* 此外，我们建议将你的网络输入课程所提供的[网页应用](http://selfdrivingcars.mit.edu/deepteslajs/)中，查看模型的预测效果。


## 评估
你的项目会由优达学城项目评审师依照[机器学习毕业项目要求](https://review.udacity.com/#!/rubrics/273/view)来评审。请确定你已完整的读过了这个要求，并在提交前对照检查过了你的项目。提交项目必须满足所有要求中每一项才能算作项目通过。
                                
                                
## 提交：
* PDF 报告文件（注意这不应该是notebook的导出，请尽量按照[模板](https://github.com/nd009/machine-learning/blob/zh-cn/projects/capstone/capstone_report_template.md)填写）
* 项目相关代码（包括从raw data开始到最终结果以及你过程中所有数据分析和作图的代码，其中分析和可视化部分建议在notebook中完成）
* 包含使用的库，机器硬件，机器操作系统，训练时间等数据的 README 文档（建议使用 Markdown ）