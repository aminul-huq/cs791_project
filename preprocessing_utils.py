import numpy as np
import cv2, glob, random, shutil, os, plistlib, pickle, math
import pandas as pd
import pydicom as dicom
from skimage.draw import polygon
import xml.etree.ElementTree as ET

def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x
    i =  0
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            i+=1
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if len(points) <= 2:
                for point in points:
                        mask[int(point[0]), int(point[1])] = i
            else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = i
    return mask


def crop(img, mask):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w], breast_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def truncation_normalization(img, mask):
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[mask==0]=0
    return normalized


def clahe(img, clip):

    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl


def synthetized_images(patient_id):

    image_path = glob.glob(os.path.join(DCM_PATH,str(patient_id)+'*.dcm'))[0]
    mass_mask = load_inbreast_mask(os.path.join(XML_PATH,str(patient_id)+'.xml'))
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array

    breast, mask, mass_mask = crop(pixel_array_numpy, mass_mask)
    normalized = truncation_normalization(breast, mask)

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)

    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    return breast, synthetized, mass_mask



class Annotation:
  
    def __init__(self, xml_path, filename, shape):

        self.xml_path = xml_path + filename + '.xml'
        self.mask     = self.create_mask_array(shape)
        self.fill_mask()


    """
  This function creates the proper contour/polygon for each ROI and stores
  the information in the corresponding position of the mask object
  """
    def fill_mask(self):
        if os.path.exists(self.xml_path):
            rois, num_rois   = self.parse_XML(self.xml_path)

        for roi in rois:
            roi_info       = self.get_roi_info(roi)
            r_poly, c_poly = self.create_polygon_lists(self.mask, roi_info["points"])
            rr, cc         = polygon(r_poly, c_poly)
            roi_channel    = self.select_mask_channel(roi_info["roi_type"])
            try:
                  self.mask[rr, cc, roi_channel] = 1
            except IndexError:
                  print(self.xml_path)


    """
  Inputs:
    -- xml_path: Path to the corresponding xml file
  Outputs:
    -- rois: Array with the ROI objects
    -- num_of_rois: Number of ROIs 
    """
    def parse_XML(self, xml_path):
        tree        = ET.parse(xml_path)
        root        = tree.getroot()       # The root of the XML file
        data        = root[0][1]           # The essential info
        rois        = data[0][5]           # Array containing the ROI objects
        num_of_rois = int(data[0][3].text) # Number of ROI objects

        return rois, num_of_rois


    """
  Inputs:
    -- img_shape: The preferred shape of the mask to be created
  Outputs:
    -- 3-dimensional numpy array of type uint8 
    """
    def create_mask_array(self,img_shape):
        return np.zeros((img_shape[0], img_shape[1], 3), dtype = np.uint8)

    def get_roi_info(self, roi):
        roi_info      = {
          "points":        roi[21],           # Array containing the points of a ROI
          "num_of_points": int(roi[17].text), # Number of points of the area
          "roi_index":     int(roi[7].text),  # Identifier of the ROI
          "roi_type":      roi[15].text       # (Mass, Calcification, other)
        }

        return roi_info



    """
  Inputs:
    -- mask: numpy object of the mask
    -- points: x-y coordinates of a ROI's points
  Outputs:
    -- r_poly: array containing the x-axis coordinates
    -- c_poly: array containing the y-axis coordinates
    """
    def create_polygon_lists(self, mask, points):
        mask_width  = mask.shape[0]
        mask_height = mask.shape[1]
        r_poly      = np.array([])
        c_poly      = np.array([])
        roi_img     = np.zeros((mask_width, mask_height), dtype=np.uint8)

        for point in points:
            temp_tuple = point.text[1:-1].split(",")
            y          = int(math.trunc(float(temp_tuple[0]))) 
            x          = int(math.trunc(float(temp_tuple[1])))
            r_poly     = np.append(r_poly, x)
            c_poly     = np.append(c_poly, y)

        return r_poly, c_poly


    """
  Input:
    -- roi_type: The type of a specific ROI, extracted from the XML file
  Output:
    -- roi_channel: The type of the ROI defines the integer value of this var
    """
    def select_mask_channel(self, roi_type):
        roi_ch = 2
        if roi_type == "Mass":
            roi_ch = 0
        elif roi_type == "Calcification":
            roi_ch = 1
        return roi_ch

























