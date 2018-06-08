from PIL import Image
from PIL import ImageEnhance
from PIL import ImageStat
from PIL import ImageDraw
import numpy as np
import cPickle
import csv
import os
import time
import random
import theano.tensor

def loadImageData(path, threshold=999999999, includeValidation=False,
                  translations=False, obstructions=False, sharpness=False, contrast=False):
    """

    load images, do data expansion if requested and return data in the correct format

    """

    images = []
    validation = []

    length = 32
    validation_ct = 10  # =n -> every n_th image is used for validation data
    counter = 0

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".csv") and file[:2] != "._":

                reader = csv.reader(open(root + "/" + file, "rb"), delimiter=';')

                rowindex = 0
                for row in reader:

                    if threshold > 0 and counter >= threshold: break

                    if rowindex > 0:

                        counter = counter + 1
                        validation_image = counter % validation_ct == 0
                        imgName = root + "/" + row[0]

                        img = Image.open(imgName)

                        new_img = img.resize((length, length), box=(float(row[3]), float(row[4]), float(row[5]), float(row[6])))

                        img_rgbRep = getRGB(new_img, length)


                        # data expansion
                        if not validation_image:

                            if sharpness:
                                enhancer = ImageEnhance.Sharpness(new_img)
                                sharpened_img_rgb = getRGB(enhancer.enhance(0.2), length)
                                images.append((sharpened_img_rgb, row[7], imgName))

                            if contrast:
                                enhancer = ImageEnhance.Contrast(new_img)
                                contrasted_img_rgb = getRGB(enhancer.enhance(0.5), length)
                                images.append((contrasted_img_rgb, row[7], imgName))

                            if translations:
                                #translation x left
                                x1, y1, x2, y2 = max(0,float(row[3])-2), max(0,float(row[4])), max(0,float(row[5])-2), max(0,float(row[6]))
                                extended_img2 = img.resize((length, length), box=(x1, y1, x2, y2))

                                #translation y up
                                x1, y1, x2, y2 = max(0,float(row[3])), max(0,float(row[4])-2), max(0,float(row[5])), max(0,float(row[6])-2)
                                extended_img3 = img.resize((length, length), box=(x1, y1, x2, y2))

                                ext_img_rgb2 = getRGB(extended_img2, length)
                                ext_img_rgb3 = getRGB(extended_img3, length)
                                images.append((ext_img_rgb2, row[7], imgName))
                                images.append((ext_img_rgb3, row[7], imgName))

                            if obstructions:
                                x0, y0, x1, y1 = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                                obstructed_img = img.resize((length, length), box=(x0, y0, x1, y1))
                                draw = ImageDraw.Draw(obstructed_img)

                                w = random.random()*0.4*length
                                h = random.random()*0.4*length
                                x = random.random()*0.9*length
                                y = random.random()*0.9*length
                                c = int(random.random() * 50 + 10)
                                draw.ellipse(fill=(c, c, c), xy=(x, y, x+w, y+h))

                                w = random.random() * 0.3 * length
                                h = random.random() * 0.3 * length
                                x = random.random() * 0.9 * length
                                y = random.random() * 0.9 * length
                                c = int(random.random() * 50 + 10)
                                draw.ellipse(fill=(c, c, c), xy=(x, y, x + w, y + h))

                                w = random.random() * 0.2 * length
                                h = random.random() * 0.2 * length
                                x = random.random() * 0.9 * length
                                y = random.random() * 0.9 * length
                                c = int(random.random() * 50 + 10)
                                draw.ellipse(fill=(c, c, c), xy=(x, y, x + w, y + h))

                                obstructed_img_rgb = getRGB(obstructed_img, length)
                                images.append((obstructed_img_rgb, row[7], imgName))

                                if counter % 2 == 0:
                                    obstructed_img = img.resize((length, length), box=(x0, y0, x1, y1))
                                    draw = ImageDraw.Draw(obstructed_img)

                                    a = random.random()*length
                                    b = random.random()*length
                                    d = random.random()*length
                                    e = random.random()*length
                                    c = int(random.random() * 50 + 10)
                                    draw.line(fill=(c, c, c), xy=(a, b, d, e), width=3)

                                    a = random.random()*length
                                    b = random.random()*length
                                    d = random.random()*length
                                    e = random.random()*length
                                    c = int(random.random() * 50 + 10)
                                    draw.line(fill=(c, c, c), xy=(a, b, d, e), width=3)

                                    a = random.random()*length
                                    b = random.random()*length
                                    d = random.random()*length
                                    e = random.random()*length
                                    c = int(random.random() * 50 + 10)
                                    draw.line(fill=(c, c, c), xy=(a, b, d, e), width=1)

                                    obstructed_img_rgb = getRGB(obstructed_img, length)
                                    images.append((obstructed_img_rgb, row[7], imgName))


                        if includeValidation and validation_image:
                            validation.append((img_rgbRep, row[7], imgName))
                        else:
                            images.append((img_rgbRep, row[7], imgName))

                        if len(images) + len(validation) % 1000 == 0:
                            print "images loaded: {0}".format(len(images) + len(validation))


                    rowindex += 1

    print "images loaded total: {0}".format(len(images) + len(validation))

    random.shuffle(images)

    if includeValidation:
        return getTuple(images), getTuple(validation)
    else:
        return getTuple(images)


def getRGB(new_img, length):
    """

    return rgb data for the provided (square) image of size length x length

    """
    img_rRep = []
    img_gRep = []
    img_bRep = []

    for a in range(length):
        for b in range(length):
            r, g, b = new_img.getpixel((a, b))
            img_rRep.append(r / 255.0)
            img_gRep.append(g / 255.0)
            img_bRep.append(b / 255.0)

    return np.concatenate((img_rRep, img_gRep, img_bRep))


def getTuple(images):
    """

    reshape the data format to a tuple holding the arrays: images, labels and filenames

    """
    xImages = []
    yClasses = []
    zNames = []

    for tupl in images:
        xImages.append(tupl[0])
        yClasses.append(tupl[1])
        zNames.append(tupl[2])

    return np.asfarray(xImages), np.asfarray(yClasses), np.asarray(zNames)





