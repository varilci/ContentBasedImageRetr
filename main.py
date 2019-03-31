import cv2
import numpy as np
import scipy.signal
import math
import datetime
import itertools
from itertools import product
import bisect
from operator import itemgetter
import sys

def splitter(angle, magnitude, less, big):
    ratio = float(angle - less) / float(abs(big - less))
    return [magnitude*(1-ratio), magnitude*ratio]

def outputCreator(file_name, bin_size, inputList, databaseList):

    print("Creating output file...")

    data = np.loadtxt(fname=file_name)
    imageList = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')
    inputFiles = np.genfromtxt(inputList, dtype=None, delimiter='\n', encoding='utf-8')

    #result = []

    outputFileName = 'parsedResult_' + str(bin_size) + '.out'

    print("Going over all the input queries...")
    with open(outputFileName, 'w') as f:
        for x in range(len(inputFiles)):

            localRes = []

            indexOfInput = np.where(imageList == inputFiles[x])

            distance = np.sqrt(np.sum(np.square(data[indexOfInput][0] - data), axis=1))

            for y in range(len(imageList)):
                test = [distance[y], imageList[y]]
                localRes.append(test)


            tempOne = inputFiles[x] + ':'
            print(tempOne)
            # tempTwo = sorted(localRes, key=lambda tt: tt[0])

            localRes.sort(key=itemgetter(0))
            test = list(itertools.chain.from_iterable(localRes))
            # tempTwo = sorted(localRes.all(), key=itemgetter(0))
            tempThree = [tempOne] + test
            tempFour = tempThree
            print(tempFour)
            for item in tempFour:
                f.write("%s" % item)
                f.write(" ")
            f.write("\n")
            # f.close()
    f.close()
    print("Output file for ranking saved as " + outputFileName)

def colorhist(bin_size, inputList, databaseList):
    print("Color histogram generation started.")
    writeArray = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')

    temp1 = list(range(0, 256, bin_size))
    temp2 = [temp1, temp1, temp1]
    hVal = list(product(*temp2))
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))
        print("----")
        # if itEr == 500:
        # print(datetime.datetime.now())

        img = cv2.imread(data[itEr])  # ZERO MEANS READ AS GRAYSCALE
        cv2.imshow('image', img)
        resized_image = img
        height, width = img.shape[:2]
        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image

        # hVal = np.arange(0, 256, 128)

        h = np.zeros(len(hVal) + 1)  # CHANGE
        for row in range(height):
            for col in range(width):
                pixelValR = img[row, col][0]
                pixelValG = img[row, col][1]
                pixelValB = img[row, col][2]
                # print(pixelValR)
                # print(pixelValG)
                # print(pixelValB)
                # print(pixelVal)
                # nearest = min(hVal, key=lambda z: abs(z - pixelVal))
                # print(nearest)

                indexR = bisect.bisect(temp1, pixelValR)
                indexG = bisect.bisect(temp1, pixelValG)
                indexB = bisect.bisect(temp1, pixelValB)

                # print(pixelValR)
                # print(pixelValG)
                # print(pixelValB)
                #
                # print(indexR)
                # print(indexG)
                # print(indexB)

                locationCalc = indexR * indexG * indexB
                # print(locationCalc)
                h[locationCalc] += 1

        normalized = h / sum(h)  # CHANGE
        # for x in range(len(h)):
        # normalized[x] = h[x]/sum(h)
        #print(len(normalized))
        writeArray.append(normalized)

    # print(datetime.datetime.now())
    file_name = 'resultColor' + str(bin_size) + '.txt'
    print("Saving the color histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)

def graycaleHist(bin_size, inputList, databaseList):
    print("Grayscale histogram generation started.")
    writeArray = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')
    print(data)
    print(data[0])
    print(datetime.datetime.now())
    # dataset/eiwJzKmWtK.jpg
    # dataset/AFjidiZyeu.jpg
    # AeRqIFREXS.jpg
    # dataset/aONepLMFlc.jpg
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))

        if itEr == 500:
            print(datetime.datetime.now())

        img = cv2.imread(data[itEr], 0)  # ZERO MEANS READ AS GRAYSCALE
        cv2.imshow('image', img)
        resized_image = img
        height, width = img.shape[:2]
        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image

        hVal = np.arange(0, 256, bin_size)

        h = np.zeros(256/bin_size)  # CHANGE
        for row in range(height):
            for col in range(width):
                pixelVal = img[row, col]
                # print(pixelVal)
                nearest = min(hVal, key=lambda z: abs(z - pixelVal))
                # print(nearest)
                if nearest > pixelVal:
                    h[np.where(hVal == nearest - bin_size)] += 1
                else:
                    h[np.where(hVal == nearest)] += 1

        normalized = h / sum(h)
        #print(normalized)
        writeArray.append(normalized)

    file_name = 'resultGrayscale' + str(bin_size) + '.txt'
    print("Saving the grayscale histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)

def gradienthist(bin_size, inputList, databaseList):
    print("Gradient histogram generation started.")
    writeArray = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')

    # dataset/eiwJzKmWtK.jpg
    # dataset/AFjidiZyeu.jpg
    # AeRqIFREXS.jpg
    # dataset/aONepLMFlc.jpg
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))

        if itEr == 500:
            print(datetime.datetime.now())

        img = cv2.imread(data[itEr], 0)  # ZERO MEANS READ AS GRAYSCALE

        resized_image = img
        height, width = img.shape[:2]
        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image

        gX_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])  # PLACES SWAPPED DUE TO CONV OPERATOR
        gY_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])

        gX = scipy.signal.convolve2d(img, gX_kernel,
                                     mode='same', boundary='fill', fillvalue=0)
        gY = scipy.signal.convolve2d(img, gY_kernel,
                                     mode='same', boundary='fill', fillvalue=0)

        # print(gX)# SIZE:480
        # print("---seper--")
        # print(gY)
        # print("-----")

        magnitude = np.sqrt(np.power(gX, 2) + np.power(gY, 2))

        angles = np.degrees(np.arctan2(gY, gX))

        # print(magnitude)# SIZE:480
        # print("---seper--")
        # print(angles)

        # TAKE ABS OF ANGLES FOR TEST

        mgntd = magnitude.astype(int)
        angls1 = np.absolute(angles)
        angls = angls1.astype(int)

        histogramGradient = []
        histogramBins = []

        if bin_size == 5:
            histogramGradient = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0]
            histogramBins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
                             115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170,
                             175]
        if bin_size == 20:
            histogramGradient = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]

        if bin_size == 60:
            histogramGradient = [0, 0, 0]
            histogramBins = [0, 60, 120]

        whatToIncrease = [0, 0]

        maxVal = 180-bin_size

        for x in range(len(magnitude)):  # 480
            for y in range(len(magnitude[0])):
                ifExists = -1
                try:
                    ifExists == histogramBins.index(angls[x][y])
                except ValueError:
                    continue
                if ifExists != -1:
                    histogramGradient[ifExists] += mgntd[x][y]
                else:
                    nearest = min(histogramBins, key=lambda z: abs(z - angls[x][y]))
                    otherIndex = 0
                    nearestIndex = histogramBins.index(nearest)
                    if angls[x][y] > nearest:
                        if nearest == maxVal:
                            whatToIncrease = splitter(angls[x][y], mgntd[x][y], nearest, otherIndex)
                        else:
                            otherIndex = nearestIndex + 1
                            whatToIncrease = splitter(angls[x][y], mgntd[x][y], nearest, histogramBins[otherIndex])
                        histogramGradient[nearestIndex] += whatToIncrease[0]
                        histogramGradient[otherIndex] += whatToIncrease[1]
                    else:
                        otherIndex = nearestIndex - 1
                        whatToIncrease = splitter(angls[x][y], mgntd[x][y], histogramBins[otherIndex], nearest)
                        histogramGradient[nearestIndex] += whatToIncrease[1]
                        histogramGradient[otherIndex] += whatToIncrease[0]


        normalized = histogramGradient / sum(histogramGradient)
        writeArray.append(normalized)

    file_name = 'resultGradient' + str(bin_size) + '.txt'
    print("Saving the gradient histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)

# ------------------------------------------------
# ------------------------------------------------
# -----------LEVELED FUNCTIONS--------------------
# ------------------------------------------------
# ------------------------------------------------

def gradienthistWLevel(bin_size, inputList, databaseList, level_no):
    writeArray = []
    writeArrayConcat = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')
    print(data)
    print(data[0])
    print(datetime.datetime.now())
    # dataset/eiwJzKmWtK.jpg
    # dataset/AFjidiZyeu.jpg
    # AeRqIFREXS.jpg
    # dataset/aONepLMFlc.jpg

    gX_kernel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])  # PLACES SWAPPED DUE TO CONV OPERATOR
    gY_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))
        if itEr == 500:
            print(datetime.datetime.now())
        img = cv2.imread(data[itEr], 0)  # ZERO MEANS READ AS GRAYSCALE

        resized_image = img
        height, width = img.shape[:2]
        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image

        sumArray = []

        if bin_size == 5:
            sumArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0]

        if bin_size == 20:
            sumArray = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        if bin_size == 60:
            sumArray = [0, 0, 0]

        concatArray = []
        maxVal = 180 - bin_size

        for cr1 in range(level_no):
            for cr2 in range(level_no):
                # print(cr1)
                # print(cr2)
                # print("--")
                y1 = int((cr1 / 2) * height)
                y2 = int(((cr1 + 1) / 2) * height)
                x1 = int((cr2 / 2) * width)
                x2 = int(((cr2 + 1) / 2) * width)
                if cr1 == 0:
                    y2 -= 1
                if cr2 == 0:
                    x2 -= 1
                # print(y1)
                # print(y2)
                # print(x1)
                # print(x2)
                thisRoundImage = img[x1: x2, y1: y2]

                gX = scipy.signal.convolve2d(thisRoundImage, gX_kernel,
                                             mode='same', boundary='fill', fillvalue=0)
                gY = scipy.signal.convolve2d(thisRoundImage, gY_kernel,
                                             mode='same', boundary='fill', fillvalue=0)

                # print(gX)# SIZE:480
                # print("---seper--")
                # print(gY)
                # print("-----")

                magnitude = np.sqrt(np.power(gX, 2) + np.power(gY, 2))

                angles = np.degrees(np.arctan2(gY, gX))

                # print(magnitude)# SIZE:480
                # print("---seper--")
                # print(angles)

                # TAKE ABS OF ANGLES FOR TEST

                mgntd = magnitude.astype(int)
                angls1 = np.absolute(angles)
                angls = angls1.astype(int)

                # mgntd = list(map(int, magnitude))
                # angls = list(map(int, np.absolute(angles)))

                histogramGradient = []
                histogramBins = []

                if bin_size == 5:
                    histogramGradient = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0]
                    histogramBins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                                     105, 110,
                                     115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170,
                                     175]
                if bin_size == 20:
                    histogramGradient = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]

                if bin_size == 60:
                    histogramGradient = [0, 0, 0]
                    histogramBins = [0, 60, 120]

                whatToIncrease = [0, 0]

                for x in range(len(magnitude)):  # 480
                    for y in range(len(magnitude[0])):
                        ifExists = -1
                        try:
                            ifExists == histogramBins.index(angls[x][y])
                        except ValueError:
                            continue
                        if ifExists != -1:
                            histogramGradient[ifExists] += mgntd[x][y]
                        else:
                            nearest = min(histogramBins, key=lambda z: abs(z - angls[x][y]))
                            otherIndex = 0
                            nearestIndex = histogramBins.index(nearest)
                            if angls[x][y] > nearest:
                                if nearest == maxVal:
                                    whatToIncrease = splitter(angls[x][y], mgntd[x][y], nearest, otherIndex)
                                else:
                                    otherIndex = nearestIndex + 1
                                    whatToIncrease = splitter(angls[x][y], mgntd[x][y], nearest,
                                                              histogramBins[otherIndex])
                                histogramGradient[nearestIndex] += whatToIncrease[0]
                                histogramGradient[otherIndex] += whatToIncrease[1]
                            else:
                                otherIndex = nearestIndex - 1
                                whatToIncrease = splitter(angls[x][y], mgntd[x][y], histogramBins[otherIndex], nearest)
                                histogramGradient[nearestIndex] += whatToIncrease[1]
                                histogramGradient[otherIndex] += whatToIncrease[0]
                sumArray += histogramGradient
                # sumArray += histogramGradient
                concatArray.append(histogramGradient)

        # plt.plot(histogramGradient)
        # plt.show()
        concatArrayFlat = np.absolute(list(itertools.chain.from_iterable(concatArray)))
        # normalized = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # APPLY L1 NORM
        print(len(sumArray))
        print(concatArrayFlat)
        # for x in range(len(normalized)):
        normalizedSum = sumArray / sum(sumArray)
        writeArray.append(normalizedSum)

        normalizedConc = concatArrayFlat / sum(concatArrayFlat)
        writeArrayConcat.append(normalizedConc)

    file_name = 'resultGradientLevel' + str(level_no) + '_' + str(bin_size) + '.txt'
    print("Saving the gradient leveled histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)

def grayscalehistWLevel(bin_size, inputList, databaseList, level_no):
    print("Grayscale histogram generation started.")
    writeArray = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')
    print(data)
    print(data[0])
    print(datetime.datetime.now())
    # dataset/eiwJzKmWtK.jpg
    # dataset/AFjidiZyeu.jpg
    # AeRqIFREXS.jpg
    # dataset/aONepLMFlc.jpg
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))

        if itEr == 500:
            print(datetime.datetime.now())

        img = cv2.imread(data[itEr], 0)  # ZERO MEANS READ AS GRAYSCALE

        resized_image = img
        height, width = img.shape[:2]

        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image

        hVal = np.arange(0, 256, bin_size)

        concatArray = []



        for cr1 in range(level_no):
            for cr2 in range(level_no):
                # print(cr1)
                # print(cr2)
                # print("--")
                y1 = int((cr1 / 2) * height)
                y2 = int(((cr1 + 1) / 2) * height)
                x1 = int((cr2 / 2) * width)
                x2 = int(((cr2 + 1) / 2) * width)
                if cr1 == 0:
                    y2 -= 1
                if cr2 == 0:
                    x2 -= 1
                # print(y1)
                # print(y2)
                # print(x1)
                # print(x2)
                thisRoundImage = img[x1: x2, y1: y2]
                height2, width2 = thisRoundImage.shape[:2]

                h = np.zeros(int(256 / bin_size))  # CHANGE
                for row in range(height2):
                    for col in range(width2):
                        pixelVal = thisRoundImage[row, col]
                        # print(pixelVal)
                        nearest = min(hVal, key=lambda z: abs(z - pixelVal))
                        # print(nearest)
                        if nearest > pixelVal:
                            h[np.where(hVal == nearest - bin_size)] += 1
                        else:
                            h[np.where(hVal == nearest)] += 1
                concatArray.append(h)

        concatArrayFlat = np.absolute(list(itertools.chain.from_iterable(concatArray)))
        normalized = concatArrayFlat / sum(concatArrayFlat)
        # print(normalized)
        writeArray.append(normalized)

    file_name = 'resultGrayscaleLevel' + str(level_no) + '_' + str(bin_size) + '.txt'
    print("Saving the grayscale leveled histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)


def colorhistWLevel(bin_size, inputList, databaseList, level_no):
    print("Color histogram generation started.")
    writeArray = []

    data = np.genfromtxt(databaseList, dtype=None, delimiter='\n', encoding='utf-8')

    temp1 = list(range(0, 256, bin_size))
    temp2 = [temp1, temp1, temp1]
    hVal = list(product(*temp2))
    for itEr in range(len(data)):
        print(itEr)
        print("out of")
        print(len(data))
        print("----")
        # if itEr == 500:
        # print(datetime.datetime.now())

        img = cv2.imread(data[itEr])  # ZERO MEANS READ AS GRAYSCALE
        cv2.imshow('image', img)
        resized_image = img
        height, width = img.shape[:2]
        if height != 640 and width != 480:
            if height != 480 and width != 640:
                if height == 640:
                    resized_image = cv2.resize(img, (480, 640))
                elif height == 480:
                    resized_image = cv2.resize(img, (640, 480))
                else:
                    resized_image = cv2.resize(img, (640, 480))
            img = resized_image


        h = np.zeros(len(hVal) + 1)  # CHANGE
        concatArray = []

        for cr1 in range(level_no):
            for cr2 in range(level_no):
                # print(cr1)
                # print(cr2)
                # print("--")
                y1 = int((cr1 / 2) * height)
                y2 = int(((cr1 + 1) / 2) * height)
                x1 = int((cr2 / 2) * width)
                x2 = int(((cr2 + 1) / 2) * width)
                if cr1 == 0:
                    y2 -= 1
                if cr2 == 0:
                    x2 -= 1
                # print(y1)
                # print(y2)
                # print(x1)
                # print(x2)
                thisRoundImage = img[x1: x2, y1: y2]
                height2, width2 = thisRoundImage.shape[:2]

                for row in range(height2):
                    for col in range(width2):
                        pixelValR = thisRoundImage[row, col][0]
                        pixelValG = thisRoundImage[row, col][1]
                        pixelValB = thisRoundImage[row, col][2]
                        # print(pixelValR)
                        # print(pixelValG)
                        # print(pixelValB)
                        # print(pixelVal)
                        # nearest = min(hVal, key=lambda z: abs(z - pixelVal))
                        # print(nearest)

                        indexR = bisect.bisect(temp1, pixelValR)
                        indexG = bisect.bisect(temp1, pixelValG)
                        indexB = bisect.bisect(temp1, pixelValB)

                        # print(pixelValR)
                        # print(pixelValG)
                        # print(pixelValB)
                        #
                        # print(indexR)
                        # print(indexG)
                        # print(indexB)

                        locationCalc = indexR * indexG * indexB
                        # print(locationCalc)
                        h[locationCalc] += 1
                concatArray.append(h)

        concatArrayFlat = np.absolute(list(itertools.chain.from_iterable(concatArray)))
        normalized = concatArrayFlat / sum(concatArrayFlat)
        # print(normalized)
        writeArray.append(normalized)

    # print(datetime.datetime.now())
    file_name = 'resultColorLevel' + str(level_no) + '_' +str(bin_size) + '.txt'
    print("Saving the color leveled histogram as " + file_name)
    np.savetxt(file_name, writeArray)

    outputCreator(file_name, bin_size, inputList, databaseList)

# ------------------------------------------------
# ------------------------------------------------
# ---------------MAIN FUNCTION--------------------
# ------------------------------------------------
# ------------------------------------------------

def main(argv):

    # file_name = 'resultColorLevel' + '1' + '_' + '2' + '.txt'
    # print("Saving the color leveled histogram as " + file_name)
    # np.savetxt(file_name, [0, 1, 2, 3])

    print("Please enter the number of the type of histogram.")
    print("(1) Gradient")
    print("(2) Grayscale")
    print("(3) 3D Color")
    print("(4) Grid Based Feature Extraction")
    histChoice = input("Enter the number: ")
    print(" ")
    print("Please enter the bin size of histogram. (Number of bins will se selected accoringly.)")
    binChoice = '0'
    if histChoice == '1':
        print("Available sizes: 5, 20 and 60")
        binChoice = input("Enter the size: ")
        print("You've selected Gradient with " + binChoice + " bin size.")
        gradienthist(int(binChoice), argv[1], argv[2])

    elif histChoice == '2':
        print("Available sizes: 1, 16, 32, 64, 128")
        binChoice = input("Enter the size: ")
        print("You've selected Grayscale with " + binChoice + " bin size.")
        graycaleHist(int(binChoice), argv[1], argv[2])

    elif histChoice == '3':
        print("Available sizes: 8, 16, 32, 64, 128")
        binChoice = input("Enter the size: ")
        print("You've selected 3D Color with " + binChoice + " bin size.")
        colorhist(int(binChoice), argv[1], argv[2])


    elif histChoice == '4':
        print("(1) Gradient with bin size of 5")
        print("(2) Grayscale with bin size of 16")
        print("(3) 3D Color with bin size of 16")
        histLevelHist = histChoice = input("Enter the number: ")
        histLevelNo = histChoice = input("Enter the level number (2 or 3): ")
        if histLevelHist == '1':
            gradienthistWLevel(5, argv[1], argv[2], int(histLevelNo))
        elif histLevelHist == '2':
            grayscalehistWLevel(16, argv[1], argv[2], int(histLevelNo))
        elif histLevelHist == '3':
            colorhistWLevel(16, argv[1], argv[2], int(histLevelNo))
        else:
            print("Please enter a valid input. Terminating the program.")

    else:
        print("Please select an available histogram type. Terminating the program.")
        return
    print("You've selected " + histChoice + " with " + binChoice + " bin size.")





if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) > 2:
        main(sys.argv)
    else:
        print("---")
        print("Please give the image database list and queryList as an input.")
