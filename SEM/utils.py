import os
import xml.etree.ElementTree as ET
import scipy.misc as misc
import numpy as np
import keras

class SEM_loader:
    """
    This class loads SEM images from a given folder including subfolders
    with corresponding labels
    Functions:  Load data of given size in the format for a conventional CNN
                Load data of given size for a siamese network
    """

    def __init__(self,lang,size,path):
        """
        Initializes data
        Input:
            lang:   which damages should be loaded
            size:   size of the images size[0] x size[1]
            path:   path to to the folder containing subfolders with images
        """

        self.lang = lang

        compressedLang = []
        for key,value in lang.iteritems():
            compressedLang.append(value)
        compressedLang = set(compressedLang)
        self.NOC = len(compressedLang)

        self.data = {}

        IGNORE = [  "/home/tom/Data/LabeledDamages/CFK_def13_rep_1_2017-08-31"]

        ##
        ## Get subfolders
        ##

        subfolders = [x[0] for x in os.walk(path)]
        subfolders = subfolders[1:]

        for ign in IGNORE:
            subfolders.remove(ign)
        # subfolders = [IGNORE]

        x = []
        y = []

        for folder in subfolders:
            print "Reading", folder
            for PATH_XML in os.listdir(folder):
                if PATH_XML.endswith(".xml"):
                    PATH_IMG = folder + "/" + PATH_XML[0:-4] + ".png"
                    img = misc.imread(PATH_IMG,flatten=True)

                    (ymax,xmax) = img.shape[0], img.shape[1]

                    assert (size[0]<xmax and size[1]<ymax), "The dimensions desired for the damage are to big"

                    tree = ET.parse(folder+"/"+PATH_XML)
                    root = tree.getroot()

                    for ob in root:
                        if ob.tag == "object":
                            ##
                            ## Check if the object is of interest
                            ##
                            INTEREST = True
                            for bbox in ob:
                                if bbox.tag == "name":
                                    if not bbox.text in lang:
                                        INTEREST = False
                                        break
                                    else:
                                        assert bbox.text in lang, "Category in xml file not in language. Problematic file: " + PATHXML
                                        category = bbox.text
                                if bbox.tag == "bndbox":
                                    x1 = int(bbox[0].text)
                                    y1 = int(bbox[1].text)
                                    x2 = int(bbox[2].text)
                                    y2 = int(bbox[3].text)

                                    centerx = (x1+x2)/2
                                    centery = (y1+y2)/2

                                    ##
                                    ## Calculate the position of the damage in the image
                                    ##

                                    x1 = centerx - size[0]/2
                                    y1 = centery - size[1]/2
                                    x2 = centerx + size[0]/2
                                    y2 = centery + size[1]/2

                                    ##
                                    ## Catch the cases in which the extract would go
                                    ## over the boundaries of the original image
                                    ##

                                    if x1<0:
                                        x1 = 0
                                        x2 = size[0]
                                    if x2>=xmax:
                                        x1 = xmax - size[0]
                                        x2 = xmax
                                    if y1<0:
                                        y1 = 0
                                        y2 = size[1]
                                    if y2>=ymax:
                                        y1 = ymax - size[1]
                                        y2 = ymax

                            if INTEREST:
                                tmp = np.zeros((size[1],size[0],1))
                                tmp[:,:,0] = img[y1:y2,x1:x2]
                                tmp = tmp*2./255. - 1.
                                x.append(tmp)
                                y.append(lang[category])

        for key, value in lang.iteritems():
            print key, ": ", y.count(value)

        print "Size of the data set: ", len(y)

        x = np.asarray(x, float)
        y = np.asarray(y, int)

        y = keras.utils.to_categorical(y, self.NOC)
        # y = keras.utils.to_categorical(y, len(lang))

        self.data = x
        self.label = y
        self.shape = size

    def getData(self, split):
        """
        Get the data and split it into training and test sets
        Input:
            split:  split ratio between training and test sets
        """

        ##
        ## Number of examples
        ##

        NOE = self.data.shape[0]

        perm = np.random.permutation(range(NOE))

        x_train = self.data[perm[0:int((1-split)*NOE)]]
        y_train = self.label[perm[0:int((1-split)*NOE)]]
        x_test = self.data[perm[int((1-split)*NOE):int(NOE)]]
        y_test = self.label[perm[int((1-split)*NOE):int(NOE)]]
        return x_train,y_train,x_test,y_test

    def getShape(self):
        return self.shape

    def getLang(self):
        return self.lang

def TrueNegatives(n_out,y_true,threshold):
    """
    Calculate the number of true negatives
    Input:
        n_out:      Output of the CNN (softmax)
        y_true:     True labels
        threshold:  Threshold above which to decide for the true case
    """

    n_out = n_out[:,0]
    y_true = y_true[:,0]

    y_pred = np.greater(n_out,threshold)

    correct = np.logical_and(y_pred, y_true)

    return y_true.shape[0]-np.count_nonzero(correct)
