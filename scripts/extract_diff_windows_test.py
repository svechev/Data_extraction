import cv2, re, sys, math, easyocr, traceback
import xlsxwriter, json, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import rcParams
from dtw import *

# EXECUTE THE SCRIPT FROM INSIDE THE "scripts" DIRECTORY 

# ==== Global constants ======
GRAY_THRESHOLD = 80
SHOW_IMAGE = False
BETWEEN = 19

# best results with window_size = 0, gray_threshold = 80 - residual mean is 15.43
# tick between labels - 19 

# ============ Image to run the code on ============
img_path = '../img/solar.ts.png'
img_name = Path(img_path).name
reader = easyocr.Reader(['en'], gpu=False)

# ========== SAVES INFORMATION FOR THE OCR TEXT =============

img_text = {}  # complete information for text, includes confidence
bbox_text = {} # information for each text and it's border box


# =============== READ TEXT FROM IMAGE WITH EASYOCR ===========
def detectText(img_name, image):
    '''
    OCR scanning of the image, save information about the scanned text
    in img_text and bbox_text if the confidence is above a certain threshold.
    Returns the image.
    '''

    img_height, img_width, _ = image.shape

    results = reader.readtext(image)


    if img_name not in img_text:
        img_text[img_name] = {}
        img_text[img_name]['TextDetections'] = []

    if img_name not in bbox_text:
        bbox_text[img_name] = []

    for (bbox, text, confidence) in results:

        # bbox is 4 points
        vertices = np.array(bbox, dtype=np.int32)
        vertices = vertices.reshape((-1, 1, 2))

        detection = {
            "DetectedText": text,
            "Confidence": float(confidence * 100),
            "Geometry": {
                "Polygon": [
                    {"X": v[0][0] / img_width, "Y": v[0][1] / img_height}
                    for v in vertices
                ]
            },
            "Type": "WORD"
        }

        img_text[img_name]["TextDetections"].append(detection)

        if confidence * 100 >= 50:
            image = cv2.fillPoly(image, [vertices], (255, 255, 255))

            x, y, w, h = cv2.boundingRect(vertices)

            bbox_text[img_name].append(
                (text, (int(x), int(y), int(w), int(h)))
            )

    return image



# DEBUG function
def show(img, title=""):
    '''
    Shows an image - used only for debugging at certain steps during
    Y-value extraction
    '''
    plt.figure(figsize=(10,6))
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()


# =============== AXES DETECTION ================================
def findMaxConsecutiveOnes(nums) -> int:
    '''
    Returns the length of the maximum sequence of consecutive ones in 
    an array of 0s and 1s.
    '''
    
    count = maxCount = 0
    
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            maxCount = max(count, maxCount)
            count = 0
                
    return max(count, maxCount)

def detectAxes(filepath, threshold=None, debug=False):
    '''
    Detects Y and X axes and returns their coordinates. Filters the
    image to black and white using a certain threshold, then finds the row
    and column with the most consecutive ones (black pixels) - they
    are the axes.
    '''
    
    if filepath is None:
        return None, None
    
    if threshold is None:
        threshold = 10
    
    COLOR_THRESHOLD = 240

    image = cv2.imread(filepath)
    height, width, channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the max-consecutive-ones for eah column in the bw image, and...
    # pick the "first" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[:, idx] < COLOR_THRESHOLD) for idx in range(width)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < width:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            break
            
        start_idx += 1
           
    yaxis = (maxindex, 0, maxindex, height)
    
    if debug:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image)

        ax[1].plot(maxConsecutiveOnes, color = 'k')
        ax[1].axhline(y = max(maxConsecutiveOnes) - 10, color = 'r', linestyle = 'dashed')
        ax[1].axhline(y = max(maxConsecutiveOnes) + 10, color = 'r', linestyle = 'dashed')
        ax[1].vlines(x = maxindex, ymin = 0.0, ymax = maxConsecutiveOnes[maxindex], color = 'b', linewidth = 4)

        plt.show()

    # Get the max-consecutive-ones for eah row in the bw image, and...
    # pick the "last" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[idx, :] < COLOR_THRESHOLD) for idx in range(height)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < height:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            
        start_idx += 1
            
    cv2.line(image, (0, maxindex), (width, maxindex),  (255, 0, 0), 2)
    xaxis = (0, maxindex, width, maxindex)

    return xaxis, yaxis
# ===========================================================


def cleanText(image_text):
    '''
    Removes some bad/fake text that might have been scanned. It's probably
    related to the old OCR scanner that the original script used,
    but I kept the function anyway 
    '''
    cleaned = []

    for item in image_text:
        try:
            text, rect = item
            textx, texty, w, h = rect
        except Exception:
            print("Skipping malformed item:", item)
            continue

        if text.strip() != 'I':
            cleaned.append((text, (textx, texty, w, h)))

    return cleaned

def getProbableLabels(image, image_text, xaxis, yaxis):
    '''
    Using the image and the already extracted text, classifies the text as 
    X-label, Y-label, X-text, Y-text, or legend text.
    Returns lists for each category that contain the classified texts
    '''
    
    y_labels = []
    x_labels = []
    legends = []
    y_text_list = []
    
    height, width, channels = image.shape
    
    (x1, y1, x2, y2) = xaxis
    (x11, y11, x22, y22) = yaxis

    XLABEL_TOL = 80 # tolerance for x-labels

    image_text = cleanText(image_text)
    

    # Check every text:
    # below x-axis -> probably an x-label or x-text
    # left of y-axis + above x-axis -> probably an y-label
    # text can be legend if it's above x-axis and is not numeric  
    for text, (textx, texty, w, h) in image_text:
        text = text.strip()

        
        cross_x = (x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)
        cross_y = (x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)


        # below x-axis -> text can be an x-label
        if (0 <= texty - y1 <= XLABEL_TOL and
            textx >= x11):
            x_labels.append((text, (textx, texty, w, h)))


        # To the left of y-axis   # and (not?) top of x-axis
        elif np.sign(cross_x) <= 0: # and np.sign(cross_y) >= 0:
            
            numbers = re.findall(r'^[+-]?\d+(?:\.\d+)?[%-]?$', text)
            if bool(numbers):
                y_labels.append((text, (textx, texty, w, h)))
            else:
                y_text_list.append((text, (textx, texty, w, h)))
                # Consider non-numeric only for legends
                legends.append((text, (textx, texty, w, h)))
            

        # Top of x-axis and to the right of y-axis
        elif (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) <= 0 and
            np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) <= 0):
            
            # Consider non-numeric only for legends
            legends.append((text, (textx, texty, w, h)))
    
    # Get the y-labels by finding the maximum
    # intersections with the sweeping line
    maxIntersection = 0
    maxList = []
    for i in range(x11):
        count = 0
        current = []
        for index, (text, rect) in enumerate(y_labels):
            if lineIntersectsRectX(i, rect):
                count += 1
                current.append(y_labels[index])
                            
        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    
    y_labels_list = maxList.copy()
    
    y_labels = []
    for text, (textx, texty, w, h) in maxList:
        y_labels.append(text)
        
    # Get the x-labels by finding the maximum
    # intersections with the sweeping line
    maxIntersection = 0
    maxList = []
    for i in range(y1 - XLABEL_TOL, height):
        count = 0
        current = []
        for index, (text, rect) in enumerate(x_labels):
            if lineIntersectsRectY(i, rect):
                count += 1
                current.append(x_labels[index])
                            
        if count > maxIntersection:
            maxIntersection = count
            maxList = current
            
    x_labels_list = maxList.copy()
    
    x_text = x_labels.copy()
    x_labels = []
    hmax = 0
    
    for text, (textx, texty, w, h) in maxList:
        x_labels.append(text)
        if texty + h > hmax:
            hmax = texty + h
    
    # Get possible x-text by moving from where we
    # left off in x-labels to the complete
    # height of the image.
    maxIntersection = 0
    maxList = []
    for i in range(hmax + 1, height):
        count = 0
        current = []
        for index, (text, rect) in enumerate(x_text):
            if lineIntersectsRectY(i, rect):
                count += 1
                current.append(x_text[index])
                            
        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    
    x_text = []
    for text, (textx, texty, w, h) in maxList:
        x_text.append(text)
    
    # Get possible legend text
    # For this, we need to search both top to
    # bottom and also from left to right.
    
    legends_and_numbers = mergeTextBoxes(legends)
    
    legends = []
    for text, (textx, texty, w, h) in legends_and_numbers:
        if not re.search(r'^([(+-]*?(\d+)?(?:\.\d+)*?[-%) ]*?)*$', text):
            legends.append((text, (textx, texty, w, h)))
    
    
    def canMerge(group, candidate):
        candText, candRect = candidate
        candx, candy, candw, candh = candRect
        
        for memText, memRect in group:
            memx, memy, memw, memh = memRect
                
            if abs(candy - memy) <= 5 and abs(candy + candh - memy - memh) <= 5:
                return True
            elif abs(candx - memx) <= 5:
                return True
                
        return False
    
    # Grouping Algorithm
    legend_groups = []
    for index, (text, rect) in enumerate(legends):

        for groupid, group in enumerate(legend_groups):
            if canMerge(group, (text, rect)):
                group.append((text, rect))
                break
        else:
            legend_groups.append([(text, rect)])
    
    #print(legend_groups)
    #print("\n\n")
    
    maxList = []
    
    # the group with the highest amount of words is classified as the legend
    if len(legend_groups) > 0:
        maxList = max(
            legend_groups,
            key=lambda g: sum(len(text.split()) for text, _ in g)
        )

    legends = []
    for text, (textx, texty, w, h) in maxList:
        legends.append(text)
        
    return image, x_labels, x_labels_list, x_text, y_labels, y_labels_list, y_text_list, legends, maxList

# helper method
def lineIntersectsRectX(candx, rect):
    (x, y, w, h) = rect
    
    if x <= candx <= x + w:
        return True
    else:
        return False

# helper method 
def lineIntersectsRectY(candy, rect):
    (x, y, w, h) = rect
    
    if y <= candy <= y + h:
        return True
    else:
        return False

# helper method
def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


# ========== GET RATIO FOR Y VALUE MATCHING =======================
def getRatio(xaxis, yaxis):
    '''
    Returns the ratio for the Y-value matching. Formula is:
    
    mean of the difference between tick values
                    divided by
    mean of the difference in pixels between ticks

    Finds the Y-label texts, calculates the means and then the ratio   
    '''
    
    list_text = []
    list_ticks = []
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape

    image_text = bbox_text[img_name]
    
    for text, (textx, texty, w, h) in image_text:
        text = text.strip()
                    
        (x1, y1, x2, y2) = xaxis
        (x11, y11, x22, y22) = yaxis
        
        # To the left of y-axis and top of x-axis
        if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
            np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == 1):
            
            # Consider numeric only for ticks on y-axis
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if bool(numbers):
                list_text.append((numbers[0], (textx, texty, w, h)))
                          
    # Get the y-labels by finding the maximum
    # intersections with the sweeping line
    maxIntersection = 0
    maxList = []
    for i in range(x11):
        count = 0
        current = []
        for index, (text, rect) in enumerate(list_text):
            if lineIntersectsRectX(i, rect):
                count += 1
                current.append(list_text[index])
                            
        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    
    # Get list of text and ticks
    list_text = []
    for text, (textx, texty, w, h) in maxList:
        list_text.append(float(text))
        list_ticks.append(float(texty + h))
        
    text_sorted = (sorted(list_text))
    ticks_sorted  = (sorted(list_ticks))
    
    ticks_diff = ([ticks_sorted[i] - ticks_sorted[i-1] for i in range(1, len(ticks_sorted))])
    text_diff = ([text_sorted[i] - text_sorted[i-1] for i in range(1, len(text_sorted))])
    print("[get text-to-tick ratio] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))
    
    # Detected text may not be perfect! Remove the outliers.
    ticks_diff = reject_outliers(np.array(ticks_diff), m=1)
    text_diff = reject_outliers(np.array(text_diff), m=1)
    print("[reject_outliers] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))
    
    normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()

    return text_sorted, normalize_ratio


# ========== MATCHING THE RATIO FOR FINAL DATA EXTRACTION ===================
def mergeRects(contours, mode='contours'):
    rects = []
    rectsUsed = []

    # Just initialize bounding rects and set all bools to false
    for cnt in contours:
        if mode == 'contours':
            rects.append(cv2.boundingRect(cnt))
        elif mode == 'rects':
            rects.append(cnt)
        
        rectsUsed.append(False)

    # Sort bounding rects by x coordinate
    def getXFromRect(item):
        return item[0]

    rects.sort(key = getXFromRect)

    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = 5
    yThr = 5

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (candxMin <= currxMax + xThr):

                    if not nearbyRectangle((candxMin, candyMin, candxMax - candxMin, candyMax - candyMin),
                                           (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin), yThr):
                        break

                    # Reset coordinates of current rect
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    break

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    
    return acceptedRects

def mergeTextBoxes(textboxes):
    rects = []
    rectsUsed = []
    
    # Just initialize bounding rects and set all bools to false
    for box in textboxes:
        rects.append(box)
        rectsUsed.append(False)

    # Sort bounding rects by x coordinate
    def getXFromRect(item):
        return item[1][0]
    
    def getYFromRect(item):
        return item[1][1]

    rects.sort(key = lambda x: (getYFromRect(x), getXFromRect(x)))
    
    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = 10
    yThr = 5

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            # Initialize current rect
            currxMin = supVal[1][0]
            currxMax = supVal[1][0] + supVal[1][2]
            curryMin = supVal[1][1]
            curryMax = supVal[1][1] + supVal[1][3]
            currText = supVal[0]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[1][0]
                candxMax = subVal[1][0] + subVal[1][2]
                candyMin = subVal[1][1]
                candyMax = subVal[1][1] + subVal[1][3]
                candText = subVal[0]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if candxMin >= currxMin and candxMin <= currxMax + xThr:

                    if not nearbyRectangle((candxMin, candyMin, candxMax - candxMin, candyMax - candyMin),
                                           (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin), yThr):
                        break

                    # Reset coordinates of current rect
                    currxMax = max(currxMax, candxMax)
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)
                    currText = currText + ' ' + candText
                    
                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    continue

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currText, (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin)])
    
    return acceptedRects

def nearbyRectangle(current, candidate, threshold):
    (currx, curry, currw, currh) = current
    (candx, candy, candw, candh) = candidate
    
    currxmin = currx
    currymin = curry
    currxmax = currx + currw
    currymax = curry + currh
    
    candxmin = candx
    candymin = candy
    candxmax = candx + candw
    candymax = candy + candh
    
    # If candidate is on top, and is close
    if candymax <= currymin and candymax + threshold >= currymin:
        return True
    
    # If candidate is on bottom and is close
    if candymin >= currymax and currymax + threshold >= candymin:
        return True
    
    # If intersecting at the top, merge it
    if candymax >= currymin and candymin <= currymin:
        return True
    
    # If intersecting at the bottom, merge it
    if currymax >= candymin and currymin <= candymin:
        return True
    
    # If intersecting on the sides or is inside, merge it
    if (candymin >= currymin and
        candymin <= currymax and
        candymax >= currymin and
        candymax <= currymax):
        return True
    
    return False

def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5

def angle_between(p1, p2):
    
    deltaX = p1[0] - p2[0]
    deltaY = p1[1] - p2[1]

    return math.atan2(deltaY, deltaX) / math.pi * 180
    
def RectDist(rectA, rectB):
    (rectAx, rectAy, rectAw, rectAh) = rectA
    (rectBx, rectBy, rectBw, rectBh) = rectB
    
    return abs(rectAx + rectAx / 2 - rectBx - rectBx / 2)

def expand(points, margin = 1):
    return np.array([
        [[points[0][0][0] - margin, points[0][0][1] - margin]],
        [[points[1][0][0] + margin, points[1][0][1] - margin]],
        [[points[2][0][0] + margin, points[2][0][1] + margin]],
        [[points[3][0][0] - margin, points[3][0][1] + margin]]])
#  =========================================================================

def filterBbox(rects, legendBox):
    text, (textx, texty, width, height) = legendBox

    filtered = []
    for rect in rects:
        (x, y, w, h) = rect

        # center alignment instead of broken edge math
        rect_cy = y + h / 2
        legend_cy = texty + height / 2

        if abs(rect_cy - legend_cy) <= 15:
            filtered.append(rect)

    filtered = mergeRects(filtered, 'rects')

    closest = None
    dist = sys.maxsize

    for rect in filtered:
        (x, y, w, h) = rect

        # horizontal proximity only
        if abs(x + w - textx) <= dist:
            dist = abs(x + w - textx)
            closest = rect

    return closest

def boxGroup(img, box):
    (x, y, w, h) = box

    image = img[y:y+h, x:x+w].reshape((h * w, 3))
    values, counts = np.unique(image, axis = 0, return_counts = True)

    # Remove white and near-by pixels
    threshold = 5
    for r in range(255 - threshold, 256):
        for g in range(255 - threshold, 256):
            for b in range(255 - threshold, 256):
                image = image[np.where((image != [r, g, b]).any(axis = 1))]

    values, counts = np.unique(image, axis = 0, return_counts = True)
                
    sort_indices = np.argsort(-counts)
    values, counts = values[sort_indices], counts[sort_indices]

    groups = []
    groupcounts = []

    for idx, value in enumerate(values):
        grouped = False

        for groupid, group in enumerate(groups):
            for member in group:
                r, g, b = member
                vr, vg, vb = value

                if (abs(vr.astype(np.int16) - r.astype(np.int16)) <= 5 and
                    abs(vg.astype(np.int16) - g.astype(np.int16)) <= 5 and
                    abs(vb.astype(np.int16) - b.astype(np.int16)) <= 5):
                        group.append(value)
                        groupcounts[groupid] += counts[idx]
                        grouped = True
                        break

            if grouped:
                break

        if not grouped:
            groups.append([value])
            groupcounts.append(counts[idx])

    groupcounts = np.array(groupcounts)
    sort_indices = np.argsort(-groupcounts)
    new_groups = [groups[i] for i in sort_indices]
    groups = new_groups
    
    return groups

def getYVal(window_size, between=0, debug=False):
    '''
    Finds the Y-values for each series in the legend
    and returns a nested dictionary

    Processing steps:
    1. Load the image and detect the X and Y axes.
    2. Extract probable X-axis labels, legends, and legend bounding boxes.
    3. Calculate the Y-axis normalization ratio (pixels -> chart values).
    4. Remove all detected text from the image to avoid interference with
       contour and color detection.
    5. Threshold the image and extract contours that may correspond to
       legend markers.
    6. Match legend text to its visual marker and determine the marker's
       dominant color group.
    7. Optionally generate virtual X-axis ticks between labeled ticks
       using linear interpolation when the 'between' parameter is greater
       than zero.
    8. For each legend:
       a. Load the original image and convert it to grayscale.
       b. Determine the vertical boundaries of the plotting area.
       c. For each X-axis tick (including virtual ticks):
          i.   Search within a horizontal window centered on the tick.
          ii.  Collect all dark pixels inside the plot area.
          iii. Group nearby pixels into clusters.
          iv.  Select the largest cluster as the detected curve.
          v.   Use the median Y-position of that cluster as the curve location.
          vi.  Convert the pixel position to a chart value using the
               normalization ratio.
          vii. Assign the computed value to the corresponding X-axis tick.
          viii.If no suitable pixels are found, assign a value of 0.
    9. Store the extracted values in a dictionary of the form:

       {
           image_name: {
               legend_name: {
                   x_label: y_value,
                   ...
               }
           }
       }
    '''
    
    yValueDict = {}    # will store the results


    # ------------------------------------------------------------------
    # Step 1: Load image and detect chart axes
    # ------------------------------------------------------------------
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape
    
    if debug:
        show(img, "Original image")

    # Axes detection
    xaxis, yaxis = detectAxes(img_path)
    
    
    for (x1, y1, x2, y2) in [xaxis]:
        xaxis = (x1, y1, x2, y2)

    for (x1, y1, x2, y2) in [yaxis]:
        yaxis = (x1, y1, x2, y2)

    print("xaxis:", xaxis)
    print("yaxis:", yaxis)


    # ------------------------------------------------------------------
    # Step 2: Extract probable X-axis labels and legends
    # ------------------------------------------------------------------
    image_text = bbox_text[img_name]

    img, x_labels, x_labels_list, _, y_labels, y_labels_list, _, legends, legendBoxes = getProbableLabels(img, image_text, xaxis, yaxis)

    actual_image = img.copy()
    
    
    try:

        # ------------------------------------------------------------------
        # Step 3: Calculate pixel-to-value normalization ratio
        # ------------------------------------------------------------------
        list_text, normalize_ratio = getRatio(xaxis, yaxis)
        
        # ------------------------------------------------------------------
        # Step 4: Remove detected text from the image
        # ------------------------------------------------------------------
        texts = img_text[img_name]['TextDetections']
        
        for text in texts:
            if text['Type'] == 'WORD' and text['Confidence'] >= 50:
                vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
                vertices = np.array(vertices, np.int32)
                vertices = vertices.reshape((-1, 1, 2))

                img = cv2.fillPoly(img, [expand(vertices, 1)], (255, 255, 255))

        if debug:
            show(img, "After removing text")

        # ------------------------------------------------------------------
        # Step 5: Threshold image and extract potential legend contours
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        if debug:
            show(threshold, "Binary threshold")

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) < 0.01 * img_height * img_width]

        contours = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
        rects = [cv2.boundingRect(contour) for contour in contours]

        # ------------------------------------------------------------------
        # Step 6: Match legends with their color markers
        # ------------------------------------------------------------------
        groups = []
        legendtexts = []
        legendrects = []
        legend_colors = []

        for box in legendBoxes:
            text, (textx, texty, width, height) = box
            bboxes = filterBbox(rects, box)
            # print("bboxes:", bboxes)

            if bboxes is not None:
                for rect in [bboxes]:
                    (x, y, w, h) = rect
                    legendrects.append(rect)
                    
                    group = boxGroup(actual_image, rect)[0]
                    group = [arr.tolist() for arr in group]
                    
                    groups.append(group)
                    legendtexts.append(text)

                    legend_colors.append(tuple(group[0])) 
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                cv2.rectangle(img, (textx, texty), (textx + width, texty + height), (255, 0, 0), 2)

        if len(groups) == 0:    
            default_color = [np.array([0, 0, 0], dtype=np.uint8)]

            groups = [default_color]
            legendtexts = legendtexts if legendtexts else legends

        # ------------------------------------------------------------------
        # Step 7: Generate virtual X-axis ticks if requested
        # ------------------------------------------------------------------
        all_ticks = sorted(
            x_labels_list,
            key=lambda x: x[1][0]   # sort by x position
        )  

        data = {}
        for legend in legends:
            data[legend] = {}
            
            # for x_label, box in x_labels_list:
            #     data[legend][x_label] = 0.0

        expanded_ticks = []

        if between == 0:
            expanded_ticks = all_ticks

        else:
            for i in range(len(all_ticks) - 1):

                left_label, left_box = all_ticks[i]
                right_label, right_box = all_ticks[i + 1]

                left_val = float(left_label)
                right_val = float(right_label)

                x1, y1, w1, h1 = left_box
                x2, y2, w2, h2 = right_box

                step = (right_val - left_val) / (between + 1)

                # create intermediate virtual ticks
                for k in range(between + 1):
                    alpha = k / (between + 1)

                    interp_x = x1 + alpha * (x2 - x1)
                    interp_y = y1 + alpha * (y2 - y1)

                    expanded_ticks.append(
                        (left_val + k * step, (interp_x, interp_y, w1, h1))
                    )

            # include last tick
            expanded_ticks.append((float(all_ticks[-1][0]), all_ticks[-1][1]))  

        x_labels_list = [
            (str(val), box) for val, box in expanded_ticks
        ]   


        # ------------------------------------------------------------------
        # Step 8: Extract values for each legend series
        # ------------------------------------------------------------------
        for i in range(len(groups)):
            legendtext = legendtexts[i]

            # --------------------------------------------------------------
            # Step 8a: Load image, convert to grayscale
            # --------------------------------------------------------------
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


            # --------------------------------------------------------------
            # Step 8b: Determine plotting boundaries
            # --------------------------------------------------------------
            # _, black_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
            y_label_ys = [box[1] for _, box in y_labels_list]
            plot_top = min(y_label_ys) + 20
            print(f"{plot_top=}")
            plot_bottom = xaxis[3] - 10
            print(f"{plot_bottom=}")

            MIN_Y_VALUE = min([int(txt) for (txt, _) in y_labels_list])

            debug_img = image.copy() 

            # Draw horizontal lines showing the top and bottom of the plot area
            cv2.line(
                debug_img,
                (0, plot_top),
                (debug_img.shape[1] - 1, plot_top),
                (0, 255, 0),   # green
                2
            )

            cv2.line(
                debug_img,
                (0, plot_bottom),
                (debug_img.shape[1] - 1, plot_bottom),
                (0, 255, 0),   # green
                2
            )


            # --------------------------------------------------------------
            # Step 8c: Process each X-axis tick
            # --------------------------------------------------------------
            for x_label, box in x_labels_list:
                x, y, w, h = box
                
                # search for dark pixels near the current x position
                x_center = int(x + w / 2)

                window = window_size
                xs = range(
                    max(0, x_center - window) - 1,
                    min(img.shape[1], x_center + window + 1) - 1
                )

                dark_pixels = gray < GRAY_THRESHOLD

                ys_all = []

                for xi in xs:
                    ys = np.where(dark_pixels[:, xi] > 0)[0]
                    ys = ys[(ys > plot_top) & (ys < plot_bottom)]
                    
                    for ypix in ys:
                        cv2.circle(debug_img, (xi, ypix), 2, (255, 0, 0), -1)   # red dot

                    if len(ys) > 0:
                        ys_all.extend(ys)

                if len(ys_all) == 0:
                    value = 0.0
                else:
                    ys_all = np.array(ys_all)

                    # sort pixels
                    ys_all = np.sort(ys_all)

                    # find densest region (simple clustering via difference)
                    diffs = np.diff(ys_all)

                    # split into clusters where gap is large
                    clusters = np.split(ys_all, np.where(diffs > 2)[0] + 1)

                    # choose largest cluster (this is usually the curve, not text)
                    main_cluster = max(clusters, key=len)
                    # main_cluster = clusters[0]

                    y_curve = int(np.median(main_cluster))

                    # convert pixel distance to value
                    pixel_height = xaxis[3] - y_curve
                    value = pixel_height * normalize_ratio + MIN_Y_VALUE

                data[legendtext][x_label] = value

            if SHOW_IMAGE:
                show(debug_img, "Detected dark pixels")

        # ------------------------------------------------------------------
        # Step 9: Store results and return dictionary
        # ------------------------------------------------------------------        
        yValueDict[img_name] = data
    
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
        sys.exit(1)
                
    return yValueDict

def process(window_size):

    # GET THE Y VALUES
    # ADD "debug = True" for displaying the image at different processing steps
    yValueDict = getYVal(window_size, between=BETWEEN, debug=False)


    if img_name in yValueDict:      

        data = yValueDict[img_name]
 
        for _, datadict in data.items():

            # Skip if no values have been detected
            vals = list(datadict.values())
            return np.array(vals[:-2])


def get_actual_data():
    # extract actual data from solar.csv
    actual_data = pd.read_csv("../data/solar.csv") # read
    actual_data = actual_data.rename(columns={actual_data.columns[-1]: "Values"}) # rename column
    # print(actual_data.columns)
    actual_data["Values"] = actual_data['Values'].astype(float) # convert to floats

    # filter data

    # the N/A value, according to the csv
    actual_data = actual_data[actual_data['Values'] != -9999.000] 

    # filter only start of year
    actual_data["Date"] = pd.to_datetime(actual_data["Date"])
    actual_data = actual_data[ (actual_data["Date"].dt.month == 1) & (actual_data["Date"].dt.day == 1)]

    # filter years between 1960 and 2018
    actual_data = actual_data[ (actual_data["Date"].dt.year >= 1960) & (actual_data["Date"].dt.year <= 2018)]

    # keep only the y values
    actual_data = np.array(actual_data["Values"])
    return actual_data

def plot_residual(residual, window_size):
    # Create a figure
    fig = plt.figure()
    fig.canvas.manager.set_window_title(f"Window size = {window_size}")

    # modify the xs so they match the year labels
    xs = np.arange(1960, 2019)

    # plot for the residual
    plt.subplot(1, 2, 1)
    plt.title("Residual plot")
    plt.xlabel("Year")
    plt.plot(xs, residual, "-or")

    # histogram for the residual
    plt.subplot(1, 2, 2)
    plt.title("Histogram of residual")
    plt.hist(residual)

    plt.show()

def plot_dtw(est_data, actual_data, window_size):
    # define the query and template
    query = est_data
    template = actual_data

    # find the best match with the canonical recursion formula
    alignment = dtw(query, template, keep_internals=True)

    print(f"\ndistance: {alignment.distance:.2f}")
    print(f"normalized distance: {alignment.normalizedDistance}")

    # Display the warping curve
    alignment.plot(type="threeway", xlab="Estimated data", ylab="Actual data")
    plt.gcf().canvas.manager.set_window_title(f"Window size = {window_size}")

    # Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw(
        query,
        template,
        keep_internals=True,
        step_pattern=rabinerJuangStepPattern(6, "c")
    ).plot(type="twoway", offset=-2)

    plt.gcf().canvas.manager.set_window_title(f"Window size = {window_size}")

    plt.show()

if __name__ == '__main__':
    # PROMPT USER FOR 3 WINDOW SIZES
    window_1 = int(input("Window 1 size: "))
    window_2 = int(input("Window 2 size: "))
    window_3 = int(input("Window 3 size: "))
    windows = [window_1, window_2, window_3]

    # OPEN THE IMAGE AND EXTRACT THE TEXT
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = detectText(img_name, image)

    est_data1 = process(window_1)
    est_data2 = process(window_2)
    est_data3 = process(window_3)

    actual_data = get_actual_data()

    residual1 = est_data1 - actual_data
    residual2 = est_data2 - actual_data
    residual3 = est_data3 - actual_data

    # print information in terminal
    print("actual_data:")
    print(actual_data)
    print()

    print("estimated data, window size", window_1)
    print(est_data1)
    print("estimated data, window size", window_2)
    print(est_data2)
    print("estimated data, window size", window_3)
    print(est_data3)
    print()

    print("residual, window size", window_1)
    print(residual1)
    print("residual, window size", window_2)
    print(residual2)
    print("residual, window size", window_3)
    print(residual3)
    print()


    mean1, abs_mean1, std1 = np.mean(residual1), np.mean(abs(residual1)) ,np.std(residual1)
    mean2, abs_mean2, std2 = np.mean(residual2), np.mean(abs(residual2)), np.std(residual2)
    mean3, abs_mean3, std3 = np.mean(residual3), np.mean(abs(residual3)), np.std(residual3)

    summary = pd.DataFrame({
    "window_size": windows,
    "mean": [mean1, mean2, mean3],
    "mean_abs": [abs_mean1, abs_mean2, abs_mean3],
    "std": [std1, std2, std3]
    })

    print(summary)

    plot_residual(residual1, window_1)
    plot_residual(residual1, window_2)
    plot_residual(residual1, window_3)

    plot_dtw(est_data1, actual_data, window_1)
    plot_dtw(est_data2, actual_data, window_2)
    plot_dtw(est_data3, actual_data, window_3)


    
    

    
    



