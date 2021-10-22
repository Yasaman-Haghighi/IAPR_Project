import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def display_image(mat, axes=None, cmap=None, hide_axis=True):
    """
    Display a given matrix into Jupyter's notebook
    
    :param mat: Matrix to display
    :param axes: Subplot on which to display the image
    :param cmap: Color scheme to use
    :param hide_axis: If `True` axis ticks will be hidden
    :return: Matplotlib handle
    """
    img = cv.cvtColor(mat, cv.COLOR_BGR2RGB) if mat.ndim == 3 else mat
    cmap= cmap if mat.ndim != 2 or cmap is not None else 'gray'
    if axes is None:
        if hide_axis:
            plt.xticks([])
            plt.yticks([])
        return plt.imshow(img, cmap=cmap)
    else:
        if hide_axis:
            axes.set_xticks([])
            axes.set_yticks([])
        return axes.imshow(img, cmap=cmap)

def preprocessing_new(frame):

    """
    :param frame: BGR image to segment
    :return: binary mask
    """
    
    # CODE HERE  

    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #create NumPy arrays from the boundaries
    lower = np.asarray([36 ,25, 25])
    upper = np.asarray([85 ,255, 255])
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
   
	  #find the colors within the specified boundaries and apply the mask
    mask = cv.inRange(hsvImage, lower, upper)

    kernel = np.ones((4,4), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return  mask

def preprocessing(frame):

    """
    :param frame: BGR image to segment
    :return: binary mask
    """
    
    # CODE HERE  

    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #create NumPy arrays from the boundaries
    lower = np.asarray([36 ,25, 25])
    upper = np.asarray([85 ,255, 255])
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
   
	  #find the colors within the specified boundaries and apply the mask
    mask = cv.inRange(hsvImage, lower, upper)

    kernel = np.ones((12,12), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return  mask

def sort_player(boxs, rects, centers):

    box_sorted = boxs.copy()
    rect_sorted = rects.copy()
    center_sorted = centers.copy()
    centers = np.concatenate(centers)

    idx = np.argmax(centers[:,1])
    box_sorted[0] = boxs[idx]
    rect_sorted[0] = rects[idx]
    center_sorted[0] = centers[idx]
    
    idx = np.argmax(centers[:,0])
    box_sorted[1] = boxs[idx]
    rect_sorted[1] = rects[idx]
    center_sorted[1] = centers[idx]
    
    idx = np.argmin(centers[:,1])
    box_sorted[2] = boxs[idx]
    rect_sorted[2] = rects[idx]
    center_sorted[2] = centers[idx]

    idx = np.argmin(centers[:,0])
    box_sorted[3] = boxs[idx]
    rect_sorted[3] = rects[idx]
    center_sorted[3] = centers[idx]
    
    return box_sorted, rect_sorted, center_sorted

def card_detection(image, mask):

    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours.sort(key=len, reverse= True)

    ranks = []
    suits = []
    centers = []
    boxs = []
    rects = []
    error = 0

    

    dealer = np.mean(contours[4], axis=0, dtype=int)

    for i in range(4):

        # find corners
        perimeter = cv.arcLength(contours[i], True)
        cnt = cv.approxPolyDP(contours[i], 5e-3 * perimeter, True)

        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        center = np.mean(box, axis=0, dtype=int)
        center = center.reshape(1, -1)
        centers.append(center)
        boxs.append(box)
        rects.append(rect[1])
        

    
    box_sorted, rect_sorted, center_sorted = sort_player(boxs, rects, centers)


    for idx, (box, rect) in enumerate(zip(box_sorted, rect_sorted)):

        img = image.copy()
        cv.drawContours(img, [box], 0, (0,255,0), 10)
        #cv.drawContours(img, center[np.newaxis, :], 0, (0,0,0), 10)

        max_corner = np.argmax(np.sum(box, axis=1))
        box = np.roll(box, 3-max_corner + idx, axis=0)

        # get width and height of the detected rectangle
        width = int(min(rect)) 
        height = int(max(rect))
        if width*height > 450000:
            error = 1

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv.warpPerspective(img, M, (width, height))
        s = np.shape(warped)
        s0 = int(s[0]/2)
        s1 = int(s[1]/2)

        rank = warped[s0-190:s0+190, s1-190:s1+190]
        ranks.append(rank)

#         s2 = int(s[0]/6.3)
#         s3 = int(s[1]/6)

#         suit = warped[s2-70:s2+70, s3-50:s3+50]
#         suits.append(suit)

        # the following 5 lines should be commented in future
        s2 = int(5.3*s[0]/6.3)
        s3 = int(5*s[1]/6)

        suit = warped[s2-70:s2+70, s3-50:s3+50]
        suit = cv.rotate(suit, cv.cv2.ROTATE_180)
        suits.append(suit)

#         fig, ax = plt.subplots(1, 3, figsize=(17, 9))
#         display_image(img, axes=ax[0])
#         ax[0].set_title('Detected card')
#         display_image(rank, axes=ax[1])
#         ax[1].set_title('Detected rank')
#         display_image(suit, axes=ax[2])
#         ax[2].set_title('Detected suit')

        plt.show()

    return ranks, suits, dealer, center_sorted, error

def find_dealer(centers, dealer_center):

    dist = np.linalg.norm(centers - dealer_center , axis = 1)
    return np.argmin(dist)