# !/usr/bin/python
# -*-coding:utf-8-*-
import time
from PIL import Image
import pytesseract
import importlib
import sys
import cv2
import re
import numpy as np
import csv


datas = []


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'w', newline='', encoding='utf-8-sig') as f:  # 追加
        writer = csv.writer(f)
        for item in datas:
            writer.writerow(item)
    print("保存文件成功，处理结束")


MIN_MATCH_COUNT = 4
importlib.reload(sys)
time1 = time.time()


def jz(pic_m, pic_path):
    img1 = cv2.imread(pic_m)
    img2 = cv2.imread(pic_path)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('C:\\Users\\pdy\\Desktop\\shenfenzhengshibie\\sfztt.jpg', g1)
    cv2.imwrite('C:\\Users\\pdy\\Desktop\\shenfenzhengshibie\\20190603020801tt.jpg', g2)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)  # 找到模板图像的特征点
    kp2, des2 = sift.detectAndCompute(g2, None)  # 找到识别的特征点

    FLANN_INDEX_KDTREE = 0  # 最近搜索
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # 最近近似匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    # 两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # reshape为(x,y)数组
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
        matchesMask = mask.ravel().tolist()
        # 使用转换矩阵M计算出img1在img2的对应形状
        h, w = cv2.UMat.get(cv2.UMat(g1)).shape  # 格式转换
        M_r = np.linalg.inv(M)
        im_r = cv2.warpPerspective(img2, M_r, (w, h))
        cv2.imwrite('hehe.jpg', im_r)
        return im_r


# 二值化算法
def binarizing(img, threshold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


# 去除干扰线算法
def depoint(img):  # input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pixdata[x, y - 1] > 245:
                count = count + 1
            if pixdata[x, y + 1] > 245:
                count = count + 1
            if pixdata[x - 1, y] > 245:
                count = count + 1
            if pixdata[x + 1, y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x, y] = 255
    return img


# 身份证识别
def identity_ocr(pic_read):
    # 身份证号码截图
    img1 = Image.open(pic_read)
    w, h = img1.size
    # 将身份证放大3倍
    out = img1.resize((w * 3, h * 3), Image.ANTIALIAS)
    region = (460 * 3, 690 * 3, 1300 * 3, 820 * 3)
    # 裁切身份证号码图片
    cropimg = out.crop(region)
    # 转化为灰度图
    img = cropimg.convert('L')
    img.save('009.png')
    # 把图片变成二值图像。
    img1 = binarizing(img, 100)
    img2 = depoint(img1)
    code = pytesseract.image_to_string(img2)
    datas.append(str(code))
    print("号码是:" + str(code))


def identity_ocr2(pic_read):
    # 身份证名字截图
    img1 = Image.open(pic_read)
    w, h = img1.size
    out = img1.resize((w, h), Image.ANTIALIAS)
    region = (245, 80, 480, 200)
    # 裁切身份证名字图片
    cropimg = out.crop(region)
    # 转化为灰度图
    img = cropimg.convert('L')
    img.save('000.png')
    # 把图片变成二值图像。
    img1 = binarizing(img, 100)
    img2 = depoint(img1)
    code = pytesseract.image_to_string(img2, lang='chi_sim')
    datas.append(str(code))
    print("名字是:" + str(code))


def identity_ocr3(pic_read):
    # 身份证性别截图
    img1 = Image.open(pic_read)
    w, h = img1.size
    out = img1.resize((w, h), Image.ANTIALIAS)
    region = (210, 180, 360, 350)
    # 裁切身份证性别图片
    cropimg = out.crop(region)
    # 转化为灰度图
    img = cropimg.convert('L')
    img.save('002.png')
    # 把图片变成二值图像。
    img1 = binarizing(img, 100)
    img2 = depoint(img1)
    code = pytesseract.image_to_string(img2, lang='chi_sim')
    datas.append(str(code))
    print("性别是:" + str(code))


def identity_ocr4(pic_read):
    # 身份证民族截图
    img1 = Image.open(pic_read)
    w, h = img1.size
    out = img1.resize((w, h), Image.ANTIALIAS)
    region = (500, 200, 640, 320)
    # 裁切身份证民族图片
    cropimg = out.crop(region)
    # 转化为灰度图
    img = cropimg.convert('L')
    img.save('004.png')
    # 把图片变成二值图像。
    img1 = binarizing(img, 100)
    img2 = depoint(img1)
    code = pytesseract.image_to_string(img2, lang='chi_sim')
    datas.append(str(code))
    print("民族是:" + str(code))


def identity_ocr5(pic_read):
    img1 = Image.open(pic_read)
    # 身份证住址截图
    w, h = img1.size
    # 将身份证放大3倍
    out = img1.resize((w * 3, h * 3), Image.ANTIALIAS)
    region = (250 * 3, 420 * 3, 850 * 3, 620 * 3)
    # 裁切身份证住址图片
    cropimg = out.crop(region)
    # 转化为灰度图
    img = cropimg.convert('L')
    img.save('006.png')
    # 把图片变成二值图像。
    img1 = binarizing(img, 100)
    img2 = depoint(img1)
    code = pytesseract.image_to_string(img2, lang='chi_sim')
    datas.append(str(code))
    print("住址是:" + str(code))


if __name__ == '__main__':
    pic_m = sys.argv[1]
    pic_path = sys.argv[2]
    pic_read = "20190525.jpg"
    #file_name = 'C:\\Users\\pdy\\Desktop\\shenfenzhengshibie\\file.csv'

    jz(pic_m, pic_path)
    identity_ocr(pic_read)
    identity_ocr2(pic_read)
    identity_ocr3(pic_read)
    identity_ocr4(pic_read)
    identity_ocr5(pic_read)
    #data_write_csv(file_name, datas)
    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')
