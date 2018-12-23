---
title: 爬虫抓取LOFTER上年度热门摄影作品
date: 2017-03-31 12:10:46
comments: false
tags:
- Python
- 爬虫
---
今天来抓一下LOFTER上[2016年度热门摄影作品](http://www.lofter.com/selection?id=1334100&type=2)的网页，并且把页面上的热门摄影作品保存在本地。
<!--more-->
Python版本:3.5.3
代码用到的库有
1.requests ：用来获取网页源代码
2.BeautifulSoup：用来解析网页获得图片链接
```Python
import requests
from bs4 import BeautifulSoup
```

下面开始写代码
首先要找到所需要抓取的网页地址，打开Chrome浏览器，找到需要抓取的网页
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-1.png)



但是仅仅用url这一个参数还不能够得到网页的代码，还需要程序伪造成浏览器进行爬去，因此还要构造一个Headers作为参数和url一起传给requests，
打开索要抓取的网页然后按F12在Network中可以找到Headers的参数
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-2.png)

找到Headers之后，就可以构造一个Headers的参数来传递给requests来伪造浏览器进行网页的抓取
```Python
	
page_url = 'http://www.lofter.com/selection?id=1334100&type=2'

Header = {
    'Accept':'',
    'Accept-Encoding':'',
    'Accept-Language':'',
    'Cache-Control':'',
    'Cookie':'',
    'Host':'',
    'Referer':'',
    'Upgrade-Insecure-Requests':'',
    'User-Agent':''
}
```
字典里的参数就是浏览器得到的参数(后来试了一下发现只要有Cookie好像就能抓到网页)，这里就不把参数贴出来了，不同的电脑参数应该是不一样的。

然后就是网页的抓取
```Python
html = requests.get(url=page_url, headers=Header)
soup = BeautifulSoup(html.text,'html.parser')
```
抓取到所需要的页面之后然后开始在页面中找到我们想要的图片的地址信息
首先找到所有照片的页面标签
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-3.png)

然后在找到单独每个照片的标签
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-4.png)

然后用BeautifulSoup找到相应的标签
```Python
pics = soup.find('div',class_='m-bd').find_all('div',class_='img')
```
打印得到的pics可以看到img标签中的src便是图片的地址
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-5.png)

然后再获得标签中的信息，并且发现src中'?'之前的便是图片的地址，因此获取图片的地址
```Python
pic_url = i.find('img')['src'].split('?')[0]
```
打印pic_url可以得到
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-6.png)

我们已经得到了图片的地址，下面就是获取图片的信息并且保存在本地
```Python
img = requests.get(pic_url)
f = open('img/' + str(count) + '.jpg', 'ab')
f.write(img.content)
f.close()
```
其中count是一个计数器并且为保存的照片命名。

最终打开文件夹，发现图片已经保存在了本地
![](http://onaxllwtn.bkt.clouddn.com/2017-03-31-7.png)

类似的方法还可以获取[年度热门绘画作品](http://www.lofter.com/selection?id=1331100&type=2),[年度热门美食作品](http://www.lofter.com/selection?id=1333100&type=2),[年度热门旅行作品](http://www.lofter.com/selection?id=1333101&type=2)等等

仔细看一下这些照片的质量还是蛮高的，完全可以拿来当壁纸用~哈哈
