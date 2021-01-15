from PIL import Image
import numpy
import math
import matplotlib.pyplot as plt
#导入你要测试的图像
im = numpy.array (Image.open ('3232/real/54-1.jpg'),'f')#将图像1数据转换为float型
im2 = numpy.array (Image.open ('3232/ours3/54-1.jpg'),'f')#将图像2数据转换为float型
print (im.shape,im.dtype)
#图像的行数
height = im.shape[0]
#图像的列数
width = im.shape[1]


#提取R通道
r = im[:,:,0]
#提取g通道
g = im[:,:,1]
#提取b通道
b = im[:,:,2]
#打印g通道数组
#print (g)
#图像1,2各自分量相减，然后做平方；
R = im[:,:,0]-im2[:,:,0]
G = im[:,:,1]-im2[:,:,1]
B = im[:,:,2]-im2[:,:,2]
#做平方
mser = R*R
mseg = G*G
mseb = B*B
#三个分量差的平方求和
SUM = mser.sum() + mseg.sum() + mseb.sum()
MSE = SUM / (height * width * 3)
PSNR = 10*math.log ( (255.0*255.0/(MSE)) ,10)

print (PSNR)
im = numpy.array (Image.open ('3232/real/54-1.jpg'))#无符号型
im2 = numpy.array (Image.open ('3232/ours3/54-1.jpg'))
plt.subplot (121)#窗口1
plt.title('origin image')
plt.imshow(im,plt.cm.gray)

plt.subplot(122)#窗口2
plt.title('rebuilt image')
plt.imshow(im2,plt.cm.gray)
plt.show()