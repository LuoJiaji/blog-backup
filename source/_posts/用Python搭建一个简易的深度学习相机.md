---
title: 用Python搭建一个简易的深度学习相机
date: 2017-12-22 13:07:58
comments: false
tags:
- Python
- TesorFlow
---
本片博客介绍了一个基于树莓派和YOLO框架的智能相机的制作过程。
<!--more-->
原文见参考资料[1]

![](/images/2017-12-22/2017-12-22-1.jpg)
亚马逊刚刚发布了[DeepLens](https://aws.amazon.com/cn/deeplens/)，这是一款智能网络摄像头，它利用机器学习来检测物体、人脸以及一些活动。DeepLens还没有上市，但智能相机的想法令人兴奋。
今天，我们将搭建一个深度学习摄像头，它可以探测到在网络摄像头图像中出现的鸟类，然后保存这只鸟的照片。最终的结果是这样的:
![](/images/2017-12-22/2017-12-22-2.jpg)
深度学习相机是一个全新的机器学习平台的开始。
DeepLens的计算能力有100GFlops的计算能力，这只是一个有趣的深度学习相机计算机所必需的计算能力的开始。在未来，这些设备将变得更加强大，并具有每秒推断数百张图像的能力。
但是谁愿意等待未来呢?


## 傻瓜相机和智能推理
我们会使用一个简易计算机(比如9美元的树莓派)，把它连接到网络摄像头，然后通过WiFi发送图像，而不是在我们的相机中直接建立一个深度学习模型。在一定延迟的情况下，我们可以构建一个和Deeplens高年相似的原型，而且更加便宜。
所以在今天的博客，我们就这么做。们用Python编写一个web服务器，将图像从树莓派发送到另一台计算机进行推理或图像检测。
![](/images/2017-12-22/2017-12-22-3.jpg)
另一台拥有更多处理能力的计算机将使用一种名为[YOLO](https://pjreddie.com/darknet/yolo/)的神经网络架构来对输入图像进行检测，并判断是否有一只鸟在摄像机的图片内。
我们将从YOLO框架开始，因为它是最快速的检测模型之一。模型的端口是基于Tensorflow，因此很容易安装和运行在许多不同的平台上。另外，如果你使用我们在这篇文章中使用的精简模型，你也可以在CPU上进行检测，无需昂贵的GPU。
回到我们的原型中。如果在相机的图像中发现了一只鸟，我们将保存这张照片以便以后进行分析。
这只是一个真正智能的深度学习相机的开始，非常基础。现在就开始着手做第一个版本的原型。

## 检测与成像
![](/images/2017-12-22/2017-12-22-4.jpg)
正如我们已经说过的，DeepLens的成像被植入了计算机。因此它可以进行基础水平检测，并通过自带的计算能力来确定这些图像是否符合你的标准。
但是，像树莓派这样的处理器，它的计算能力无法做到实时检测。因此，我们将使用另一台计算机来推断图像。
在这个例子中，使用了一个简单的Linux计算机，它带有一个摄像头和wifi接入(树莓派3和一个便宜的网络摄像头)，并服务于用于图像推断的深度学习计算机。
这是一个很好的方案，因为它允许在野外使用许多便宜的相机，并且在同一个地方的台式机上进行计算。

## 摄像头图像处理方案
如果你不想使用树莓派的摄像头，同样可以在树莓派上安装OpenCV 3。作为旁注，我必须禁用CAROTENE的编译，以便在我的树莓派上获得3.3.1。你可能需要做同样的事情。完成之后，我们只需要安装Flask的web服务器，这样我们就可以从网络摄像头中加载图像了。
使用了[Miguel Grinberg](https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited)的优秀[网络摄像头服务器代码](https://github.com/miguelgrinberg/flask-video-streaming)作为基础，并创建了一个简单的jpg端点，而不是一个动态的jpeg端点:
```python
#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response

# uncomment below to use Raspberry Pi camera instead
# from camera_pi import Camera

# comment this out if you're not using USB webcam
from camera_opencv import Camera

app = Flask(__name__)

@app.route('/')
def index():
    return "hello world!"

def gen2(camera):
    """Returns a single image frame"""
    frame = camera.get_frame()
    yield frame

@app.route('/image.jpg')
def image():
    """Returns a single current image for the webcam"""
    return Response(gen2(Camera()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
```
如果你想要使用树莓派的视频摄像头，请确确保使用`from camera_pi`代码。并且注释掉`from camera_opencv`
你可以通过`python3 app.py`来运行程序，或者使用`gunicorn`，就像Miguel Grinberg的文章中提到的那样。它只是利用米格尔的出色的相机管理来关闭摄像头，当没有请求的时候，它还可以管理线程，如果我们有不止一台机器对来自网络摄像头的图像进行推断的话。
一旦我们在树莓派上启动它，我们就可以测试并确保服务器首先发现它的IP地址，然后尝试通过我们的web浏览器来实现它。
URL应该类似于http://192.168.1.4:5000 image.jpg:
![](/images/2017-12-22/2017-12-22-5.jpg)

## 从相机服务器中提取图像并进行推断
现在我们已经有了一个端点来加载网络摄像头的当前图像，我们可以构建脚本来获取并运行这些图像的推断。
我们将使用`requests`模块，一个伟大的Python库，用于从url中抓取文件；以及Darkflow，在Tensorflow上实现的YOLO模型。
不幸的是，不能通过pip来安装Darkflow，所以我们需要复制代码到本地，然后在本地编译和安装，并进行图像的检测和推理。
在安装了Darkflow之后，我们还需要下载我们将要使用的YOLO版本的权重和模型。
在这个例子中，使用了YOLO v2微型网络，因为我想在一台较慢的计算机上运行我的检测和推理的程序，使用CPU，而不是GPU。这个微小的网络与完整的YOLO v2模型相比，它的精度要低一些。
此外，我们还需要在检测计算机上安装Pillow、numpy和OpenCV模块。
最后，可以编写代码来运行探测程序了:
```python
from darkflow.net.build import TFNet
import cv2

from io import BytesIO
import time
import requests
from PIL import Image
import numpy as np

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1}

tfnet = TFNet(options)

birdsSeen = 0
def handleBird():
    pass

while True:
    r = requests.get('http://192.168.1.11:5000/image.jpg') # a bird yo
    curr_img = Image.open(BytesIO(r.content))
    curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

    result = tfnet.return_predict(curr_img_cv2)
    print(result)
    for detection in result:
        if detection['label'] == 'bird':
            print("bird detected")
            birdsSeen += 1
            curr_img.save('birds/%i.jpg' % birdsSeen)
    print('running again')
    time.sleep(4)
```
到此，我们就有了一个非常基本的第一版图片检测的代码。我们可以在控制台看到树莓派的检测，我们也可以看到每一个被看到的鸟类被保存在我们的硬盘上。
之后，我们可以运行一个程序来对YOLO已经检测到的鸟类的图像进行标记。

## 调整参数
需要注意的一点是，我们创建的选项字典中的阈值键。这个阈值表示我们需要检测的物体的置信度。
出于测试的目的，我将它设置为0.1。但是这个门槛的低会给我们带来很多错误的信息。更糟糕的是，我们用于检测的精简YOLO模型比真正的YOLO模型的准确度要低一些，因此我们将会有一些错误的检测。
降低或提高阈值可以提高或降低模型的总输出，这取决于想要构建的内容。在这个例子中，倾向于更多的假阳性结果，更喜欢得到更多的鸟的图片而不是更少的。可以根据需要调整这些参数以满足特定的需要。

## 等待检测结果
让鸟飞到鸟食器里花了很长时间。我想我在后院放了几只鸟，在几小时内把食物放出来。相反，它花了几天时间。松鼠一直在吃我不吃的食物，在最初的几天里，我几乎没有看到天空中有一只鸟。最后，把二只鸟喂食器放在一个更加显眼的地方。通过这个，我终于得到了一些图片，就像在文章开头的那些图片一样。

## 下一步
这篇文章的代码同样可以在[Github](https://github.com/burningion/poor-mans-deep-learning-camera)上获得。这篇文章是一系列课程的开始，我将使用深度学习相机尝试与鸟类互动。

参考资料:
1. [Building a Deep Learning Camera with a Raspberry Pi and YOLO](https://www.makeartwithpython.com/blog/poor-mans-deep-learning-camera/)