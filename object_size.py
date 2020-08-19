#import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#read image
image_orig = cv2.imread("image.png")
image_to_process = image_orig.copy()
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
#面積を探す
cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#輪郭ごとにループしてboxで囲む
for c in cnts:
    if cv2.contourArea(c) < 1000:
        continue
    if cv2.contourArea(c) > 2500:
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

#コロニー数と処理した画像を出力
print("{} colonies".format(counter))
cv2.imwrite("image_to_process.png", image_to_process)

"""
ToDO:
コロニーのサイズを枠の横に表示させる
"""



"""
Old! 後で消す
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#construct the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
help = "width of the left-most object in the image(in inches)")
args = vars(ap.parse_args())

#load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

#perform edge detection, then perform a dilation + erosion to
#close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort the contours from left-to-right and initialize the 
#'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
	#if the contour is not sufficiently large, ignore it　ここの大きさを変えることでコロニー検知
	if cv2.contourArea(c) < 100:
		continue

	#compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	#order the points in the contour such that they appear
	#in top-left, top-right, bottom-right, and bottom-left
	#order, then draw the outline of the rotated bounding
	#box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	#loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x),int(y)), 5, (0, 0, 255), -1)

	#unpack the ordered bounding box, then compute the midpoint
	#between the top-left and top-right coordinates, followed by 
	#the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	#compute the midpoint between the top-left and top-right points,
	#followed by the midpoint between the top-right and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	#draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	#draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), 
	(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), 
	(255, 0, 255), 2)

	#compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	#if the pixels per metric has not been initialized, then
	#compute it as the ration of pixels to supplied metric
	#(in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]
	
	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	#draw the object sizes on the image
	cv2.putText(orig, "{:.1f}in".format(dimA),
	(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
	(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)

	#show the output image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
"""