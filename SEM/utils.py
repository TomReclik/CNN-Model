import os
import xml.etree.ElementTree as ET
import scipy.misc as misc

class SEM_loader:
    """
    This class loads SEM images from a given folder including subfolders
    with corresponding labels
    Functions:  Load data of given size in the format for a conventional CNN
                Load data of given size for a siamese network
    """

    def __init__(self,size,path):
        """
        Initializes data
        Input:
            size:   size of the images size[0] x size[1]
            path:   path to to the folder containing subfolders with images
        """

        lang = {"Martensite":0, "Interface":1, "Boundary":2, "Evolved":3, "NotClassified":4}

        self.data = {}

        ##
        ## Get subfolders
        ##

        subfolders = [x[0] for x in os.walk(path)]
        subfolders = subfolders[1:]

        x = []
        y = []

        for folder in subfolders:
            for PATH_XML in os.listdir(folder):

                PATH_IMG = folder + PATHXML[0:-4] + ".png"
                img = misc.imread(PATH_IMG,flatten=True)

                (ymax,xmax) = img.shape

                assert (size[0]<xmax and size[1]<ymax), "The dimensions desired for the damage excerts are to big"

                tree = ET.parse(folder+PATHXML)
                root = tree.getroot()

                for ob in root:
                    if ob.tag == "object":
                        for bbox in ob:
                            if bbox.text == "name":
                                if bbox.text == "Artefact" or bbox.text == "Inclusion" or bbox.text == "FerriteBridge":
                                    break
                                else:
                                    assert category in language, "Category in xml file not in language. Problematic file: " + PATHXML
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

                                x.append(img[x1:x2,y1:y2])
                                y.append(lang(category))

        x = np.array(x, float)
