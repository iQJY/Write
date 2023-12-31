import numpy as np
import struct
import os
import cv2


class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'  # 大端格式
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

        self._imgNums = 0
        self._LabelNums = 0

    def getImage(self):
        """
        将MNIST的二进制文件转换成像素特征数据
        """
        binfile = open(self._filename, 'rb')  # 以二进制方式打开文件
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic, self._imgNums, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        print('image nums: %d' % self._imgNums)
        for i in range(self._imgNums):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            images.append(imgVal)
        return np.array(images), self._imgNums

    def getLabel(self):
        """
        将MNIST中label二进制文件转换成对应的label数字特征
        """
        binFile = open(self._filename, 'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, self._LabelNums = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(self._LabelNums):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY, imgNums):
        """
        根据生成的特征和数字标号，输出图像
        """
        output_txt = self._outpath + '/img.txt'
        output_file = open(output_txt, 'a+')

        m, n = np.shape(arrX)
        # 每张图是28*28=784Byte
        for i in range(imgNums):
            img = np.array(arrX[i])
            img = img.reshape(28, 28)
            # print(img)
            outfile = str(i) + "_" + str(arrY[i]) + ".bmp"
            # print('saving file: %s' % outfile)

            txt_line = outfile + " " + str(arrY[i]) + '\n'
            output_file.write(txt_line)
            cv2.imwrite(self._outpath + '/' + outfile, img)
        output_file.close()


if __name__ == '__main__':
    # 二进制文件路径，需要修改，和自己的相对应
    trainfile_X = 'D:\\桌面\\手写数字识别\\mnist\\train-images.idx3-ubyte'
    trainfile_y = 'D:\\桌面\\手写数字识别\\mnist\\train-labels.idx1-ubyte'
    testfile_X = 'D:\\桌面\\手写数字识别\\mnist\\t10k-images.idx3-ubyte'
    testfile_y = 'D:\\桌面\\手写数字识别\\mnist\\t10k-labels.idx1-ubyte'

    # 加载mnist数据集
    train_X, train_img_nums = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X, test_img_nums = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    # 以下内容是将图像保存到本地文件中
    path_trainset = "D:\\桌面\\手写数字识别\\try1\\train"
    path_testset = "D:\\桌面\\手写数字识别\\try1\\test"
    if not os.path.exists(path_trainset):
        os.mkdir(path_trainset)
    if not os.path.exists(path_testset):
        os.mkdir(path_testset)
    DataUtils(outpath=path_trainset).outImg(train_X, train_y, int(train_img_nums / 10))  # /10是只转换十分之一，用于测试
    DataUtils(outpath=path_testset).outImg(test_X, test_y, int(test_img_nums / 10))
