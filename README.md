LK光流法的基本原理
1. 简介
   光流法旨在基于图像的像素的移动估计出特征点的位移矢量，从而估计出特征点在当前时刻的位置。
2.工作原理
  LK算法是一种两帧差分的光流估计算法，其基本思想基于以下三个假设
  （1）亮度恒定：场景中目标图像的像素看起来在帧到帧移动是不发生改变。对于灰度图像（对于彩色图像同适用）这意味着像素的灰度值不会随着帧的跟踪改变。
  （2）时间持续性（微小移动）：图像上相机的移动随时间变化缓慢。实际上，这意味着时间的变化不会引起像素位置的剧烈变化，这样像素的灰度值才能对位置求对应的偏导数。
  （3）空间一致性：场景中相同表面的相邻点具有相似的运动，并且其投影到图像平面上的距离也比较近。
   在基于第三点空间一致性可以计算出某一点的速度矢量，从而估计出像素在下一时刻的位置
