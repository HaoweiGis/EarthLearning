import cv2
import numpy as np    
infile = r'C:\Users\hp\Desktop\picture1\Capture_00002\Capture_00001.png'


#Load the image as grayscale image
image = cv2.imread(infile,0)

convertedImage = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)


cv2.imshow('image',convertedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imwrite(r'C:\Users\hp\Desktop\picture1\Capture_00002\Capture_00001_test2.png', convertedImage)


# cv2.imshow('image',convertedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread(infile,1)
# bw = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
# resized = cv2.resize(bw, (0,0), fx=0.3, fy=0.3)
# cv2.imshow('image',resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import numpy as np
# imrows = int(9600)
# imcols = int(6422)
# imsize = imrows*imcols
# with open(infile, "rb") as rawimage:
#     img = np.fromfile(rawimage, np.dtype('u1'), imsize).reshape((imrows, imcols))[:,:6421]
#     colour = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

# cv2.imshow('image',colour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# imagePath = '/path/to/image'
# imageRaw = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

# rgb = cv2.cvtColor(imageRaw, cv2.COLOR_BAYER_BG2BGR)
# cv2.imwrite('rgb.png', rgb)

# import gdal

# src = gdal.Open(r'C:\Users\hp\Desktop\picture1\Capture_00002\Capture_00001.png')
# print(src.RasterCount)
# print(src.ReadAsArray().shape)
# print(src.ReadAsArray())