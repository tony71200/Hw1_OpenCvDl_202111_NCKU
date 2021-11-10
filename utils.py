import sys
import os
import glob
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d

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
    def waitkey():
        cv.waitKey()

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
        self.image = cv.cvtColor(self.load_image(), cv.COLOR_BGR2GRAY)
        self.Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.Gy = np.flip(self.Gx.T, axis=0)
        self.sobel_X = None
        self.sobel_Y = None

    @staticmethod
    def __gaussian_kernels(size = 5, sigma= 1):
        upper = size - 1
        lower = - int(size/2)
        y, x = np.mgrid[lower: upper, lower:upper]

        kernel = (1 / (2* np.pi * sigma ** 2)) * np.exp(- (x**2 + y**2) / (2*sigma**2))

        return kernel 

    def gaussian_blur(self):
        image = self.image.copy()
        gaussian_kernel = self.__gaussian_kernels(3,1)
        
        post_gf_conv = convolve2d(image, gaussian_kernel, mode='same', 
                                boundary='fill', fillvalue=0)
        post_gf_conv = np.round(post_gf_conv)
        post_gf_conv = post_gf_conv.astype(np.uint8)
        self.showimage(image, "Origin")
        self.showimage(post_gf_conv, "3.1 Gaussian Blur")
        return post_gf_conv

    def sobelX(self, show_image = False):
        image = self.image.copy()
        sobel_X = convolve2d(image, self.Gx, mode='same', boundary='fill', fillvalue=0 )
        if show_image:
            self.showimage(np.absolute(sobel_X).astype(np.uint8), "3.2 Sobel X")
        return sobel_X

    def sobelY(self, show_image= False):
        image = self.image.copy()
        sobel_Y = convolve2d(image, self.Gy, mode='same', boundary='fill', fillvalue=0 )
        if show_image:
            self.showimage(np.absolute(sobel_Y).astype(np.uint8), "3.3 Sobel Y")
        return sobel_Y

    def magnitude(self):
        if self.sobel_X is None and self.sobel_Y is None:
            self.sobel_X = self.sobelX()
            self.sobel_Y = self.sobelY()
        gradient_magnitude = np.sqrt(np.square(self.sobel_X) + np.square(self.sobel_Y))
        gradient_magnitude *= 255.0/gradient_magnitude.max()
        self.showimage(gradient_magnitude.astype(np.uint8), '3.4 Magnitude')

class Q4(Q1):
    def __init__(self, path):
        super().__init__(path)
        self.image = self.load_image()

    @staticmethod
    def waitkey():
        cv.waitKey()
    
    def resize(self, size:tuple):
        image = self.image.copy()
        resize_image = cv.resize(image, size)
        self.showimage(resize_image, "Resized Image")
        return resize_image

    def translations(self, tX = 0, tY = 0):
        img = self.image.copy()
        h, w = img.shape[:2]
        M = np.array([[1,0,tX], [0,1, tY]], dtype=np.float32)
        translation_image = cv.warpAffine(img, M, (400, 300))
        self.showimage(translation_image, "After Translation")
        return translation_image

    def rotate(self, angle= 0, scale = 1.0):
        img = self.image.copy()
        h, w = img.shape[:2]
        center = (w/2, h/2)
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        rotate_image = cv.warpAffine(img, rotate_matrix, (400,300))
        self.showimage(rotate_image, "After Rotate")
        return rotate_image

    def shearing(self):
        img = self.image.copy()
        h, w = img.shape[:2]
        old_location = np.float32([[50,50],[200,50],[50,200]])
        new_location = np.float32([[10,100],[200,50],[100,250]])

        # # shear_matrix = np.array([[1, shX, 0], [shY,1,0], [0,0,1]], dtype=np.float32)
        shear_matrix = cv.getAffineTransform(old_location, new_location)
        sheared_img = cv.warpAffine(img, shear_matrix, (400,300))
        self.showimage(sheared_img, "Shearing Image")
        return sheared_img


if __name__ == "__main__":
    q4 = Q4("./Q4_Image/SQUARE-01.png")
    q4.shearing()
    