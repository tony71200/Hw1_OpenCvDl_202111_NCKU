import sys
import os
import glob
import cv2 as cv
import numpy as np

class Q1():

    def __init__(self, path):
        self.path = path
        self.q1_1 = False
        self.q1_2 = False

    def load_image(self, path_image = None, show_image= False):
        if path_image is None:
            image = cv.imread(self.path)
        else:
            image = cv.imread(path_image)
        if show_image:
            self.showimage(image, "1.1 Load Image")
            cv.waitKey(0)
        self.q1_1 = True
        return image

    @staticmethod    
    def showimage(image, win_name):
        cv.namedWindow(win_name, cv.WINDOW_GUI_EXPANDED)
        cv.imshow(win_name, image)

    def color_seperation(self, show_image = False):
        if not self.q1_1:
            self.image = self.load_image()
        image = self.image.copy()
        self.seperation = cv.split(image)
        self.q1_2 = True
        if show_image:
            Blue_image = np.zeros_like(image, dtype=np.uint8)
            Blue_image[:, :, 0] = self.seperation[0]
            Green_image = np.zeros_like(image, dtype=np.uint8)
            Green_image[:, :, 1] = self.seperation[1]
            Red_image = np.zeros_like(image, dtype=np.uint8)
            Red_image[:, :, 2] = self.seperation[2]
            self.showimage(Blue_image, "1.2 Color Seperation: Blue")
            self.showimage(Green_image, "1.2 Color Seperation: Green")
            self.showimage(Red_image, "1.2 Color Seperation: Red")
            cv.waitKey(0)
        pass

    def color_traformations(self):
        if not self.q1_2:
            self.color_seperation()
        image = self.image.copy()
        gray1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.showimage(gray1, "1.3 Color Traformation: Gray")
        B,G,R = self.seperation
        gray2 = (R + G + B)/3
        gray2 = np.array(gray2, dtype=np.uint8)
        self.showimage(gray1, "1.3 Color Traformation: I = (R+G+B)/3")
        cv.waitKey(0)
        pass

    def blending(self):
        dog_1 = self.load_image("./Q1_Image/Dog_Strong.jpg")
        dog_2 = self.load_image("./Q1_Image/Dog_Weak.jpg")

        max_weight = 100
        default_value = 50
        window_name = "1.4 Blending Image"
        trackbarvalue = "Blend"

        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

        def blendImage(*args):
            blend_factor = args[0]/max_weight

            blended_image = cv.addWeighted(dog_1, 1.0 - blend_factor, 
                dog_2, blend_factor, 0)

            cv.imshow(window_name, blended_image)
        cv.createTrackbar(trackbarvalue, window_name, default_value, max_weight, blendImage)
        cv.waitKey(0)
        pass

class Q2(Q1):
    def __init__(self, path):
        super().__init__(path)
        self.image = self.load_image()

    def median_filter(self, image = None, kernel_size = 5):
        if image is None:
            image = self.image.copy()
        blur_image = cv.medianBlur(image, kernel_size)
        self.showimage(blur_image, "Median Filter: {kernel}x{kernel}".format(kernel = kernel_size))
    

    def gaussian_filter(self, image = None, kernel_size = 5):
        if image is None:
            image = self.image.copy()
        blur_image = cv.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
        self.showimage(blur_image, "Gaussian Blur")
        cv.waitKey(0)

    def bilateral_filter(self, kernel_size, gammaColor, gammaSpace, image = None):
        if image is None:
            image = self.image.copy()
        blur_image = cv.bilateralFilter(image, kernel_size, gammaColor, gammaSpace)
        self.showimage(blur_image, "Bilateral Blur")
        cv.waitKey(0)

    @staticmethod
    def call_waitkey(time=0):
        cv.waitKey(time)

class Q3(Q1):
    def __init__(self, path):
        super().__init__(path)
        self.image = self.load_image()

    def gaussian_blur(self):
        image = self.image.copy()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        Gaussian_kernel = [[]]

    def sobelX(self):
        image = self.image.copy()
        

    def sobelY(self):
        pass

    def magnitude(self):
        pass

class Q4(Q1):
    def __init__(self, path):
        super().__init__(path)
        self.image = self.load_image()
    
    def resize(self, size:tuple):
        image = self.image.copy()
        resize_image = cv.resize(image, size)
        self.showimage(resize_image, "Resized Image")
        cv.waitKey()
        return resize_image

    def translations(self, tX = 0, tY = 0):
        img = self.image.copy()
        h, w = img.shape[:2]
        M = np.array([[1,0,tX], [0,1, tY]], dtype=np.float32)
        translation_image = cv.warpAffine(img, M, (400, 300))
        self.showimage(translation_image, "After Translation")
        cv.waitKey()
        return translation_image

    def rotate(self, angle= 0, scale = 1.0):
        img = self.image.copy()
        h, w = img.shape[:2]
        center = (w/2, h/2)
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        rotate_image = cv.warpAffine(img, rotate_matrix, (400,300))
        self.showimage(rotate_image, "After Rotate")
        cv.waitKey()
        return rotate_image

    def shearing(self):
        img = self.image.copy()
        h, w = img.shape[:2]
        old_location = np.float32([[50,50],[200,50],[50,200]])
        new_location = np.float32([[10,100],[200,50],[100,250]])
        src = order_points(old_location)
        dst = order_points(new_location)

        # # shear_matrix = np.array([[1, shX, 0], [shY,1,0], [0,0,1]], dtype=np.float32)
        shear_matrix = cv.getPerspectiveTransform(src, dst)
        sheared_img = cv.warpPerspective(img,shear_matrix,(400, 300), flags=cv.INTER_LINEAR)
        self.showimage(sheared_img, "Shearing Image")
        cv.waitKey()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

if __name__ == "__main__":
    q4 = Q4("./Q4_Image/SQUARE-01.png")
    q4.shearing()
    