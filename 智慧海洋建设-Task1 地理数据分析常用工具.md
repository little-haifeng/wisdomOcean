# 智慧海洋建设-Task1 地理数据分析常用工具

## [一 ，shapely的安装及使用](https://shapely.readthedocs.io/en/stable/project.html)

## 1.shapely的安装
~~~python
 #查看jupyter notebook  python编译器版本号 
import sys  
print(sys.version)
print(sys.executable)
~~~
[Shapely下载跳转链接](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414185001484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

```python
# 将下载好的whl文件剪切到Anaconda\Scripts文件夹里面
# 博主本人下载的是Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
pip install  F:\Anaconda\Scripts\Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
```
![3](https://img-blog.csdnimg.cn/20210414185918885.png)

```python
# 也能在搜索该路径下搜索框中输入cmd
pip install Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl
```

## 2.Shapely的使用

[参考资料](https://mp.weixin.qq.com/s/DvTxxRGpA2JF9OdGdsYphw)  
[官方资料](https://shapely.readthedocs.io/en/stable/project.html)

> #### 空间数据模型  
> 1.point类型对应的方法在Point类中。curve类型对应的方法在LineString和LinearRing类中。surface类型对应的方法在Polygon类中。
> 2.point集合对应的方法在MultiPoint类中，curves集合对应的反方在MultiLineString类中，surface集合对应的方法在MultiPolygon类中。
>
> #### 几何对象的一些功能特性 Point、LineString和LinearRing有一些功能非常有用。
> - 几何对象可以和numpy.array互相转换。
>
> - 可以求线的长度(length)，面的面积（area)，对象之间的距离(distance),最小最大距离(hausdorff_distance),对象的bounds数组(minx,
> miny, maxx, maxy)
>
> - 可以求几何对象之间的关系：相交(intersect)，包含(contain)，求相交区域(intersection)等。
>
> - 可以对几何对象求几何中心(centroid),缓冲区(buffer),最小旋转外接矩形(minimum_rotated_rectangle)等。
>
> - 可以求线的插值点(interpolate),可以求点投影到线的距离(project),可以求几何对象之间对应的最近点(nearestPoint)
>
> - 可以对几何对象进行旋转(rotate)和缩放(scale)

```python
from shapely import geometry as geo
from shapely import wkt 
from shapely import ops
import numpy as np 
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing
```
### Point
```python
# point有三种赋值方式，具体如下
point = geo.Point(0.5,0.5)    #坐标
point_2 = geo.Point((0,0))  
point_3 = geo.Point(point)
# 其坐标可以通过coords或x，y，z得到
print(list(point_3.coords))
print(point_3.x)
print(point_3.y)
#批量进行可视化
geo.GeometryCollection([point,point_2])
print(np.array(point))#可以和np.array进行互相转换
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414191150286.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
### LineStrings  
>class LineString(coordinates)  
>LineStrings构造函数传入参数是2个或多个点元组
```python
#代码示例
arr=np.array([(0,0), (1,1), (1,0)])
line = geo.LineString(arr) #等同于 line = geo.LineString([(0,0), (1,1), (1,0)]) 

print ('两个几何对象之间的距离:'+str(geo.Point(2,2).distance(line)))#该方法即可求线线距离也可以求线点距离  线与点的最短距离
print ('两个几何对象之间的hausdorff_distance距离:'+str(geo.Point(2,2).hausdorff_distance(line)))#该方法求得是点与线的最长距离
print('该几何对象的面积:'+str(line.area))
print('该几何对象的坐标范围:'+str(line.bounds))
print('该几何对象的长度:'+str(line.length))
print('该几何对象的几何类型:'+str(line.geom_type))  
print('该几何对象的坐标系:'+str(list(line.coords)))
center = line.centroid #几何中心
geo.GeometryCollection([line,center])  #将线与中心点一起画
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414191836557.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

**几何对象的最小外接矩形**
```python
bbox = line.envelope  #envelope可以求几何对象的最小外接矩形
geo.GeometryCollection([line,bbox]) 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414192700183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)


**最小旋转外接矩形**
```python
rect = line.minimum_rotated_rectangle #最小旋转外接矩形
geo.GeometryCollection([line,rect])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414192740712.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

**插值**
```python
pt_half = line.interpolate(0.5,normalized=True) #插值 这条线段的1/2中点
geo.GeometryCollection([line,pt_half])
ratio = line.project(pt_half,normalized=True) # project()方法是和interpolate方法互逆的  这条线段的插值插到中间部分
print(ratio)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414192913769.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414193158375.png)
[**DouglasPucker算法**](https://blog.csdn.net/foreverling/article/details/78066632)

```python
line1 = geo.LineString([(0,0),(1,-0.2),(2,0.3),(3,-0.5),(5,0.2),(7,0)])
line1_simplify = line1.simplify(0.4, preserve_topology=False)  #Douglas-Pucker算法
print(line1)
print(line1_simplify)
line1_simplify
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414193250881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
**加粗**
```python
buffer_with_circle = line1.buffer(0.2)  #端点按照半圆扩展
geo.GeometryCollection([line1,buffer_with_circle])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414193324856.png)
#### LinearRings
>
>class LinearRing(coordinates)
>LineStrings构造函数传入参数是2个或多个点元组
>
>元组序列可以通过在第一个和最后一个索引中传递相同的值来显式关闭。否则，将第一个元组复制到最后一个索引，从而隐式关闭序列。
>与LineString一样，元组序列中的重复点是允许的，但可能会导致性能上的损失，应该避免在序列中设置重复点。
>

```python
ring = geo.polygon.LinearRing([(0, 0), (1, 1), (1, 0)])    # geo.LineString
print(ring.length)#相比于刚才的LineString的代码示例，其长度现在是3.41，是因为其序列是闭合的   
print(ring.area)  #不是闭合图形
geo.GeometryCollection([ring])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414194000653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
#### Polygon  

> class Polygon(shell[, holes=None])  
> Polygon接受两个位置参数，第一个位置参数是和LinearRing一样，是一个有序的point元组。第二个位置参数是可选的序列，其用来指定内部的边界

```python
from shapely.geometry import Polygon
polygon1 = Polygon([(0, 0), (1, 1), (1, 0)])
ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)]
polygon2 = Polygon(ext, [int])
print(polygon1.area)  
print(polygon1.length)
print(polygon2.area)#其面积是ext的面积减去int的面积
print(polygon2.length)#其长度是ext的长度加上int的长度
print(np.array(polygon2.exterior))  #外围坐标点
geo.GeometryCollection([polygon2])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414194149342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
#### 几何对象关系

> 一个几何对象特征分别有interior、boundary和exterior。下面的叙述直接用内部、边界和外部等名词概述
>
> 1.object.contains(other) 如果object的外部没有其他点，或者至少有一个点在该object的内部，则返回True a.contains(b)与 b.within(a)的表达是等价的

```python
coords = [(0, 0), (1, 1)]
print(LineString(coords).contains(Point(0.5, 0.5)))#线与点的关系
print(LineString(coords).contains(Point(1.0, 1.0)))#因为line的边界不是属于在该对象的内部，所以返回是False
polygon1 = Polygon( [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
print(polygon1.contains(Point(1.0, 1.0)))#面与点的关系
#同理这个contains方法也可以扩展到面与线的关系以及面与面的关系
geo.GeometryCollection([polygon1,Point(1.0, 1.0)])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414194350155.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

```python
2.object.crosses(other)  
如果一个object与另一个object是内部相交的关系而不是包含的关系，则返回True  
3.object.disjoint(other)   
如果该对象与另一个对象的内部和边界都不相交则返回True  
4. object.intersects(other)  
如果该几何对象与另一个几何对象只要相交则返回True。  
5. object.convex_hull  
返回包含对象中所有点的最小凸多边形（凸包）
```

```python
print( LineString(coords).crosses(LineString([(0, 1), (1, 0)]))) 
print(Point(0, 0).disjoint(Point(1, 1)))
print( LineString(coords).intersects(LineString([(0, 1), (1, 0)])))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021041419445663.png)

绘制凸包
```python
# 在下图中即为在给定6个point之后求其凸包，并绘制出来的凸包图形
points1 = geo.MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
hull1 = points1.convex_hull
geo.GeometryCollection([hull1,points1])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414194855647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

交集
```python
geo.GeometryCollection([hull1,polygon1])
```

```python
# object.intersection  返回对象与对象之间的交集
polygon1 = Polygon( [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
hull1.intersection(polygon1)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414195014725.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414195045251.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

并集
```python
#返回对象与对象之间的并集
hull1.union(polygon1)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414195211963.png)


补集
```python
hull1.difference(polygon1) #面面补集
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414195127596.png)

> 另外还有一些非常有用但是不属于某个类方法的函数，如有需要可以在官网查阅
> - ops.nearest_points 求最近点
> - ops.split 分割线
> - ops.substring  求子串
> - affinity.rotate 旋转几何体
> - affinity.scale 缩放几何体
> - affinity.translate 平移几何体


# geopandas安装及使用



> GeoPandas提供了地理空间数据的高级接口，它让使用python处理地理空间数据变得更容易。GeoPandas扩展了[pandas](GeoPandas的目标是使使用python处理地理空间数据更容易。它结合了熊猫和shape的功能，提供了熊猫的地理空间操作和shape的多个几何图形的高级接口。GeoPandas使您能够轻松地在python中进行操作，否则将需要空间数据库(如PostGIS)。)使用的数据类型，允许对几何类型进行空间操作。几何运算由[shapely](https://shapely.readthedocs.io/en/stable/)执行。Geopandas进一步依赖[fiona](https://fiona.readthedocs.io/en/latest/)进行文件访问，依赖[matplotlib](http://matplotlib.org/)进行绘图。
> geopandas和pandas一样，一共有两种数据类型：
> - GeoSeries
> - GeoDataFrame 它们继承了pandas数据结构的大部分方法。这两个数据结构可以当做地理空间数据的存储器，shapefile文件的pandas呈现。  
>
> Shapefile文件用于描述几何体对象：点，折线与多边形。例如，Shapefile文件可以存储井、河流、湖泊等空间对象的几何位置。除了几何位置，shp文件也可以存储这些空间对象的属性，例如一条河流的名字，一个城市的温度等等。
## [geopandas安装](https://www.jianshu.com/p/21de55e84d90)
geopandas安装需要其他依赖包

```python
pip install Shapely
pip install pyproj
pip install Fiona
pip install GDAL
conda install -c conda-forge geopandas
```

## geopandas的使用

```python
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))#read_file方法可以读取shape文件，转化为GeoSeries和GeoDataFrame数据类型。
world.plot()#将GeoDataFrame变成图形展示出来，得到世界地图
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021041420133370.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

```python
#根据每一个polygon的pop_est不同，便可以用python绘制图表显示不同国家的人数
fig, ax = plt.subplots(figsize=(9,6),dpi = 400)
world.plot('pop_est',ax = ax,legend = True)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414201410340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

## Folium
[官方文档](https://python-visualization.github.io/folium/index.html)

```python
import folium
import os
#首先，创建一张指定中心坐标的地图，这里将其中心坐标设置为北京。zoom_start表示初始地图的缩放尺寸，数值越大放大程度越大
m=folium.Map(location=[39.9,116.4],zoom_start=10)
m
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414201536308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414201628685.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
绘制热力图
```python
import folium
import numpy as np
from folium.plugins import HeatMap
#先手动生成data数据，该数据格式由[纬度，经度，数值]构成
data=(np.random.normal(size=(100,3))*np.array([[1,1,1]])+np.array([[48,5,1]])).tolist()
# data
m=folium.Map([48,5],tiles='stamentoner',zoom_start=6)
HeatMap(data).add_to(m)
m 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414201718200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

# Kepler.gl  

> Kepler.gl与folium类似，也是是一个图形化的数据可视化工具，基于Uber的大数据可视化开源项目deck.gl创建的demo
> app。目前支持3种数据格式：CSV、JSON、GeoJSON。
>
> Kepler.gl[官网](https://kepler.gl/)提供了可视化图形案例，分别是Arc（弧）、Line（线）、Hexagon（六边形）、Point（点）、Heatmap（等高线图）、GeoJSON、Buildings（建筑）。

```python
import pandas as pd 
import geopandas as gpd
from pyproj import Proj 
from keplergl import KeplerGl
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import shapely
import numpy as np
from datetime import datetime  
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimSun']    # 指定默认字体为新宋体。
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时 负号'-' 显示为方块和报错的问题。
```

```python
#获取文件夹中的数据
def get_data(file_path,model):
    assert model in ['train', 'test'], '{} Not Support this type of file'.format(model)
    paths = os.listdir(file_path)
#     print(len(paths))
    tmp = []
    for t in tqdm(range(len(paths))):
        p = paths[t]
        with open('{}/{}'.format(file_path, p), encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                tmp.append(line.strip().split(','))
    tmp_df = pd.DataFrame(tmp)
    if model == 'train':
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    else:
        tmp_df['type'] = 'unknown'
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    tmp_df['lat'] = tmp_df['lat'].astype(float)
    tmp_df['lon'] = tmp_df['lon'].astype(float)
    tmp_df['speed'] = tmp_df['speed'].astype(float)
    tmp_df['direction'] = tmp_df['direction'].astype(float)#如果该行代码运行失败，请尝试更新pandas的版本
    return tmp_df
# 平面坐标转经纬度，供初赛数据使用
# 选择标准为NAD83 / California zone 6 (ftUS) (EPSG:2230)，查询链接：https://mygeodata.cloud/cs2cs/
def transform_xy2lonlat(df):
    x = df['lat'].values
    y = df['lon'].values
    p=Proj('+proj=lcc +lat_1=33.88333333333333 +lat_2=32.78333333333333 +lat_0=32.16666666666666 +lon_0=-116.25 +x_0=2000000.0001016 +y_0=500000.0001016001 +datum=NAD83 +units=us-ft +no_defs ')
    df['lon'], df['lat'] = p(y, x, inverse=True)
    return df  
#修改数据的时间格式
def reformat_strtime(time_str=None, START_YEAR="2019"):
    """Reformat the strtime with the form '08 14' to 'START_YEAR-08-14' """
    time_str_split = time_str.split(" ")
    time_str_reformat = START_YEAR + "-" + time_str_split[0][:2] + "-" + time_str_split[0][2:4]
    time_str_reformat = time_str_reformat + " " + time_str_split[1]
#     time_reformat=datetime.strptime(time_str_reformat,'%Y-%m-%d %H:%M:%S')
    return time_str_reformat
#计算两个点的距离
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km * 1000

def compute_traj_diff_time_distance(traj=None):
    """Compute the sampling time and the coordinate distance."""
    # 计算时间的差值
    time_diff_array = (traj["time"].iloc[1:].reset_index(drop=True) - traj[
        "time"].iloc[:-1].reset_index(drop=True)).dt.total_seconds() / 60

    # 计算坐标之间的距离
    dist_diff_array = haversine_np(traj["lon"].values[1:],  # lon_0
                                   traj["lat"].values[1:],  # lat_0
                                   traj["lon"].values[:-1], # lon_1
                                   traj["lat"].values[:-1]  # lat_1
                                   )

    # 填充第一个值
    time_diff_array = [time_diff_array.mean()] + time_diff_array.tolist()
    dist_diff_array = [dist_diff_array.mean()] + dist_diff_array.tolist()
    traj.loc[list(traj.index),'time_array'] = time_diff_array
    traj.loc[list(traj.index),'dist_array'] = dist_diff_array
    return traj 

#对轨迹进行异常点的剔除
def assign_traj_anomaly_points_nan(traj=None, speed_maximum=23,
                                   time_interval_maximum=200,
                                   coord_speed_maximum=700):
    """Assign the anomaly points in traj to np.nan."""
    def thigma_data(data_y,n): 
        data_x =[i for i in range(len(data_y))]
        ymean = np.mean(data_y)
        ystd = np.std(data_y)
        threshold1 = ymean - n * ystd
        threshold2 = ymean + n * ystd
        judge=[]
        for data in data_y:
            if (data < threshold1)|(data> threshold2):
                judge.append(True)
            else:
                judge.append(False)
        return judge
    # Step 1: The speed anomaly repairing
    is_speed_anomaly = (traj["speed"] > speed_maximum) | (traj["speed"] < 0)
    traj["speed"][is_speed_anomaly] = np.nan

    # Step 2: 根据距离和时间计算速度
    is_anomaly = np.array([False] * len(traj))
    traj["coord_speed"] = traj["dist_array"] / traj["time_array"]
    
    # Condition 1: 根据3-sigma算法剔除coord speed以及较大时间间隔的点
    is_anomaly_tmp = pd.Series(thigma_data(traj["time_array"],3)) | pd.Series(thigma_data(traj["coord_speed"],3))
    is_anomaly = is_anomaly | is_anomaly_tmp
    is_anomaly.index=traj.index
    # Condition 2: 轨迹点的3-sigma异常处理
    traj = traj[~is_anomaly].reset_index(drop=True)
    is_anomaly = np.array([False] * len(traj))

    if len(traj) != 0:
        lon_std, lon_mean = traj["lon"].std(), traj["lon"].mean()
        lat_std, lat_mean = traj["lat"].std(), traj["lat"].mean()
        lon_low, lon_high = lon_mean - 3 * lon_std, lon_mean + 3 * lon_std
        lat_low, lat_high = lat_mean - 3 * lat_std, lat_mean + 3 * lat_std

        is_anomaly = is_anomaly | (traj["lon"] > lon_high) | ((traj["lon"] < lon_low))
        is_anomaly = is_anomaly | (traj["lat"] > lat_high) | ((traj["lat"] < lat_low))
        traj = traj[~is_anomaly].reset_index(drop=True)
    return traj, [len(is_speed_anomaly) - len(traj)]
```

```python
df=get_data(r'hy_round1_train_20200102','train')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414202200848.png)
数据时间的修改
```python
df=transform_xy2lonlat(df)  ## 平面坐标转经纬度，供初赛数据使用
df['time']=df['time'].apply(reformat_strtime)   ##修改数据的时间格式将时间换成年月日
df['time']=df['time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
```

线性插值 异常值处理
```python
#这一个cell的代码不用运行，DF.csv该数据已经放到了github上面给出的附件数据里面

#对轨迹进行异常点剔除，对nan值进行线性插值
ID_list=list(pd.DataFrame(df['ID'].value_counts()).index)
DF_NEW=[]
Anomaly_count=[]
for ID in tqdm(ID_list):
    df_id=compute_traj_diff_time_distance(df[df['ID']==ID])  #去除异常值后的数据
    df_new,count=assign_traj_anomaly_points_nan(df_id)   #异常值cout
    df_new["speed"] = df_new["speed"].interpolate(method="linear", axis=0)
    df_new = df_new.fillna(method="bfill")
    df_new = df_new.fillna(method="ffill")
    df_new["speed"] = df_new["speed"].clip(0, 23)
    Anomaly_count.append(count)#统计每个id异常点的数量有多少
    DF_NEW.append(df_new)
DF=pd.concat(DF_NEW)
```

```python
#读取github的数据
DF=pd.read_csv('DF.csv')
```

> 由于数据量过大，如果直接将轨迹异常点剔除的数据用kepler.gl展示则在程序运行时会出现卡顿，或者无法运行的情况，此时可尝试利用geopandas对数据利用douglas-peucker算法进行简化。有效简化后的矢量数据可以在不损失太多视觉感知到的准确度的同时，带来巨大的性能提升。

```python
#douglas-peucker案例，由该案例可以看出针对相同ID的轨迹，可以先用geopandas将其进行简化和数据压缩
line= shapely.geometry.LineString(np.array(df[df['ID']=='11'][['lon','lat']]))
ax=gpd.GeoSeries([line]).plot(color='red')
ax = gpd.GeoSeries([line]).simplify(tolerance=0.000000001).plot(color='blue', 
                                                        ax=ax,
                                                        linestyle='--')
LegendElement = [plt.Line2D([], [], color='red', label='简化前'),
                 plt.Line2D([], [], color='blue', linestyle='--', label='简化后')]

# 将制作好的图例映射对象列表导入legend()中，并配置相关参数
ax.legend(handles = LegendElement, 
          loc='upper left', 
          fontsize=10)
# ax.set_ylim((-2.1, 1))
# ax.axis('off')
print('化简前数据长度：'+str(len(np.array(gpd.GeoSeries([line])[0]))))
print('化简后数据长度：'+str(len(np.array(gpd.GeoSeries([line]).simplify(tolerance=0.000000001)[0]))))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414202445263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)
保存文件
```python
#定义数据简化函数。即通过shapely库将经纬度转换成LineString格式
#然后放入GeoSeries数据结构中并进行简化，最后再将所有数据放入GeoDataFrame中
def simplify_dataframe(df):
    line_list=[]
    for i in tqdm(dict(list(df.groupby('ID')))):
        line_dict={}
        lat_lon=dict(list(df.groupby('ID')))[i][['lon','lat']]
        line=shapely.geometry.LineString(np.array(lat_lon))
        line_dict['ID']=dict(list(df.groupby('ID')))[i].iloc[0]['ID']
        line_dict['type']=dict(list(df.groupby('ID')))[i].iloc[0]['type']  
        line_dict['geometry']=gpd.GeoSeries([line]).simplify(tolerance=0.000000001)[0]
        line_list.append(line_dict)
    return gpd.GeoDataFrame(line_list)
df_gpd_change=simplify_dataframe(DF)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414202533471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

```python
df_gpd_change=pd.read_pickle(r'C:\Users\33309\Desktop\智慧海洋\数据集\df_gpd_change.pkl') 
map1=KeplerGl(height=800)#zoom_start与这个height类似，表示地图的缩放程度
map1.add_data(data=df_gpd_change,name='data')
#当运行该代码后，下面会有一个kepler.gl使用说明的链接，可以根据该链接进行学习参考
map1
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414204010964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

# GeoHash

> 在对于经纬度进行数据分析和特征提取时常用到的是GeoHash编码，该编码方式可以将地理经纬度坐标编码为由字母和数字所构成的短字符串，它具有如下特性：
>
> 1. 层级空间数据结构，将地理位置用矩形网格划分，同一网格内地理编码相同
> 2. 只要编码长度足够长，可以表示任意精度的地理位置坐标
> 3. 编码前缀匹配的越长，地理位置越邻近。
>
> 那么GeoHash算法是怎么对经纬度坐标进行编码的呢？总的来说，它采用的是二分法不断缩小经度和纬度的区间来进行二进制编码，最后将经纬度分别产生的编码奇偶位交叉合并，再用字母数字表示。举例来说，对于一个坐标116.29513,40.04920的经度执行算法：
>
> 1. 将地球经度区间[-180,180]二分为[-180,0]和[0,180]，116.29513在右区间，记1；
> 1. 将[0,180]二分为[0,90]和[90,180]，116.29513在右区间，记1；
> 1. 将[90,180]二分为[90,135]和[135,180]，116.29513在左区间，记0；
> 1. 递归上述过程（左区间记0，右区间记1）直到所需要的精度，得到一串二进制编码11010 01010 11001。
>  同理将地球纬度区间[-90,90]根据纬度40.04920进行递归二分得到二进制编码10111 00011 11010，接着生成新的二进制数，它的偶数位放经度，奇数位放纬度，得到11100 11101 00100 01101 11110 00110，最后使用32个数字和字母（字母去掉a、i、l、o这4个）进行32进制编码，即先将二进制数每5位转化为十进制28 29 4 13 30 6，然后对应着编码表进行映射得到wy4ey6。


```python
# reference: https://github.com/vinsci/geohash
def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)
```

```python
#调用Geohash函数
DF[DF['ID']==1].apply(lambda x: geohash_encode(x['lat'], x['lon'], 7), axis=1)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210414204445350.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ2NDU4MTY0,size_16,color_FFFFFF,t_70)

> GeoHash的主要价值在于将二维的经纬度坐标信息编码到了一维的字符串中，在做地理位置索引时只需要匹配字符串即可，便于缓存、信息压缩。在使用大数据工具（例如Spark）进行数据挖掘聚类时，GeoHash显得更加便捷和高效。
>
> 但是使用GeoHash还有一些注意事项：
>
> 1. 由于GeoHash使用Z形曲线来顺序填充空间的，而Z形曲线在拐角处会有突变，这表现在有些相邻的网格的编码前缀比其他网格相差较多，因此利用前缀匹配可以找到一部分邻近的区域，但同时也会漏掉一些。
> 1. 一个网格内部所有点会共用一个GeoHash值，在网格的边缘点会匹配到可能较远但是GeoHash值相同的点，而本来距离较近的点却没有匹配到。这种问题可以这样解决：适当增加GeoHash编码长度，并使用周围的8个近邻编码来参与，因为往往只使用一个GeoHash编码可能会有严重风险！