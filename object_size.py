#import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(a, b):
	return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

#read image
image_orig = cv2.imread("image.png")
image_to_process = image_orig.copy()
image_contours = image_orig.copy()
counter = 0

#画像のカラーを反転 GBR形式
image_to_process = (255-image_to_process)
image_to_process_1 = image_to_process.copy() #テスト出力用

#画像をグレースケールに変換
image_gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)  ###input
image_gray = cv2.GaussianBlur(image_gray, (7, 7), 0)

#シャーレの文字（黒）をバックグラウンドの色に書き換える
image_gray = np.where(image_gray >= 135, 100, image_gray)
image_gray = np.where(image_gray <= 110, 0, image_gray)
image_gray = np.where((image_gray < 135) & (image_gray > 110), 255, image_gray)
cv2.imwrite("image_gray.png", image_gray)

#エッジを探す
image_edged = cv2.Canny(image_gray, 0, 255)
image_edged = cv2.dilate(image_edged, None, iterations=1)
image_edged = cv2.erode(image_edged, None, iterations=1)
cv2.imwrite("image_edged.png", image_edged)

#輪郭を探す
cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)

#輪郭ごとにループしてboxで囲む
for c in cnts:
    if cv2.contourArea(c) < 1000: # 画像サイズに対して変更する必要がある
        continue
    if cv2.contourArea(c) > 2500:　# 画像サイズに対して変更する必要がある
        continue
    
    #boxで囲む
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    
    box = perspective.order_points(box)
    cv2.drawContours(image_to_process, [box.astype("int")], -1, (0, 255, 0), 2)
    
    #loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(image_to_process, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    #compute the Convex Hull of the contour
    hull = cv2.convexHull(c) # cv2.convexHull(c)とは？？？？
    cv2.drawContours(image_contours, [hull], 0, (0,255,0),1)
    
    counter += 1

	#unpack the orderd bounding box, then compute the midpoint
    #between the top-left and top-right coordinates, followed by
    #the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    
    
    #draw the midpoints on the image
    cv2.circle(image_contours, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image_contours, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    
    #draw the object sizes on the image
    #計測値を書く
    cv2.putText(image_contours, "{:.1f}".format(dA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                 0.65, (255, 255, 255), 2)


#コロニー数と処理した画像を出力
print("{} colonies".format(counter))
cv2.imwrite("image_to_process.png", image_to_process)