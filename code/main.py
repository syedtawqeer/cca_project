#for resizing/reshaping  the image 
import cv2 as cv
import os

files = os.listdir("capscum_org2")

for file in files:
    img = cv.imread(f"capscum_org2/{file}")
    # print(img.shape)
    half_img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # cv.imshow("image", half_img)
    cv.imwrite(f"capscum_org3/{file}", half_img)
    # break
# cv.waitKey(0)
