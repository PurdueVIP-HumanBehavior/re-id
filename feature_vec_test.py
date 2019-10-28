import opt
import distancemetrics
import vectorgenerator

from PIL import Image
import os
import numpy
import matplotlib.pyplot as plt
import random

# gallery = galleries.options[args.gallery]()
# detector = detectors.options[args.detector]()

distance = distancemetrics.options[opt.args.distance]

def distmetric(feats1, feats2):
    sum = 0
    for i in range(0, len(feats1)):
        vec1 = feats1[i]
        vec2 = feats2[i]
        vec1 = vec1.cpu().detach().numpy()
        vec2 = vec2.cpu().detach().numpy()
        vec1 = vec1 / numpy.linalg.norm(vec1)
        vec2 = vec2 / numpy.linalg.norm(vec2)
        dist = distance(vec1, vec2)
        sum = sum + dist
    return sum


def hist(path):
    vecgen = vectorgenerator.options[opt.args.vectgen]()

    # path = '../crops/moiz'
    imgs = os.listdir(path)
    imgsfi = [img for img in imgs if 'f' in img]
    imgsse = [img for img in imgs if 's' in img]

    minlen = min([len(imgsfi), len(imgsse)])
    imgsfi = imgsfi[0:minlen]
    imgsse = imgsse[0:minlen]


    vals = list()
    for ind in range(0, minlen):
        print(imgsfi[ind], imgsse[ind])
        img1 = Image.open(os.path.join(path, imgsfi[ind]))
        img2 = Image.open(os.path.join(path, imgsse[ind]))
        vec1 = vecgen.getVect(img1)
        vec2 = vecgen.getVect(img2)
        dist = distmetric(vec1, vec2)
        vals.append(dist)

    vals = [val.item() for val in vals]
    plt.hist(vals, bins=10)
    plt.show()

def compare(imgs1, imgs2):
    vecgen = vectorgenerator.options[opt.args.vectgen]()

    minlen = min([len(imgs1), len(imgs2)])
    imgs1 = imgs1[0:minlen]
    imgs2 = imgs2[0:minlen]

    vals = list()
    for ind in range(0, minlen):
        print(imgs1[ind], imgs2[ind])
        img1 = Image.open(imgs1[ind])
        img2 = Image.open(imgs2[ind])
        vec1 = vecgen.getVect(img1)
        vec2 = vecgen.getVect(img2)
        dist = distmetric(vec1, vec2)
        vals.append(dist)

    vals = [val.item() for val in vals]
    plt.hist(vals, bins=10)
    plt.show()


##################################################
# code for randomly partitioning images of same person into two halves
##################################################
def org(path):
    imgs = os.listdir(path)
    random.shuffle(imgs)
    firsthf = 0
    seconhf = 0
    for img in imgs:
        dec = random.randint(1, 100)
        if dec > 50:
            name = "f%03d.jpg" % firsthf
            firsthf = firsthf + 1
        else:
            name = "s%03d.jpg" % seconhf
            seconhf = seconhf + 1
        os.rename(os.path.join(path, img), os.path.join(path, name))

def getvec2out(path, vecgen):
    moiz = Image.open(path)
    return vecgen.getVect2(moiz)


if __name__ == "__main__":
    # path = '../crops/sunjeon'
    # #org(path)
    # #hist(path)
    ################################################################################
    # path1 = '../crops/ethan'
    # path2 = '../crops/sunjeon'
    # imgs1 = os.listdir(path1)
    # imgs1 = [os.path.join(path1, part) for part in imgs1]
    # imgs2 = os.listdir(path2)
    # imgs2 = [os.path.join(path2, part) for part in imgs2]
    # compare(imgs1, imgs2)
    ##############################################################################3
    # path1 = '../crops/moiz/fi000.jpg'
    # path2 = '../crops/moiz/fi004.jpg'
    # path3 = '../crops/ethan/f028.jpg'
    vecgen = vectorgenerator.options[opt.args.vectgen]()
    # out1 = getvec2out(path1, vecgen)
    # out2 = getvec2out(path2, vecgen)
    # out3 = getvec2out(path3, vecgen)
    # res1 = numpy.dot(out1, numpy.transpose(out2))
    # res2 = numpy.dot(out1, numpy.transpose(out3))
    # print(res1)
    # print(res2)
    #################################################################################
    moizpath = '../crops/moiz/fi004.jpg'
    ethanpath = '../crops/ethan/f028.jpg'
    sunpath = '../crops/sunjeon/f014.jpg'

    ids = ['moiz', 'ethan', 'sun']
    gal = [getvec2out(moizpath, vecgen), getvec2out(ethanpath, vecgen), getvec2out(sunpath, vecgen)]

    moizs = ['../crops/moiz/'+img for img in os.listdir('../crops/moiz')]
    ethans = ['../crops/ethan/' + img for img in os.listdir('../crops/ethan')]
    suns = ['../crops/sunjeon/' + img for img in os.listdir('../crops/sunjeon')]
    lowestnum = min([len(moizs), len(ethans), len(suns)])
    moizs = moizs[0:lowestnum]
    ethans = ethans[0:lowestnum]
    suns = suns[0:lowestnum]

    tests = {
        'moiz': moizs,
        'ethan': ethans,
        'sun': suns
    }

    data = list()
    samedists = list()
    diffs = list()

    success = 0
    total = 0
    for key, imgs in tests.items():
        for val in imgs:
            # out = getvec2out(val, vecgen)
            # dists = [numpy.average(numpy.dot(out, numpy.transpose(out2))) for out2 in gal]
            # for val in dists:
            #     data.append(val)
            # if max(dists) > .74:
            #     indexoi = dists.index(max(dists))
            #     if indexoi < len(ids):
            #         persid = ids[dists.index(max(dists))]
            #         if persid == key:
            #             success = success + 1
            #
            #     sameid = ids.index(key)
            #     samedists.append(dists[sameid])
            #     for i, val in enumerate(dists):
            #         if i != sameid:
            #             diffs.append(val)
            # else:
            #     gal.append(out)
            out = getvec2out(val, vecgen)
            dists = [numpy.average(numpy.dot(out, numpy.transpose(out2))) for out2 in gal]
            id = ids[dists.index(max(dists))]
            if id == key:
                success = success + 1


            total = total + 1
            print(total)

    print(success/total)
    print(len(gal))
    # plt.hist(samedists, bins=10)
    # plt.show()
    # plt.hist(diffs, bins=10)
    # plt.show()
    # plt.hist(data, bins=10)
    # plt.show()
