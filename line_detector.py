import cv2
import numpy as np
import matplotlib.pylab as plt
from shapely.geometry import LineString


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    cv2.imshow("mask", mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def apply_mask(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # White color mask
    lower_threshold = np.uint8([160, 160, 160])
    # lower_threshold = np.uint8([230, 230, 230])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = white_mask  # cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def drow_the_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (50, 100, 200), thickness=2)
    return img


# image = cv2.imread('init.png')

def detect_lines(image, region_of_interest_vertices):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blur_image = cv2.medianBlur(image, 1)
    gray_image = apply_mask(blur_image)
    cv2.imshow("gray_image", gray_image)

    canny_image = cv2.Canny(gray_image, 75, 150)
    # cv2.imshow("canny_image", canny_image)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32), )
    cv2.imshow("cropped_img", cropped_image)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=50)
    return lines


def line_intersection_dection(bounding_line, line_target_line):
    if line_target_line is None:
        return
    if bounding_line is None:
        return
    # bounding_line = (([0, 0, 1900, 1200],),)
    result = {}
    for key in line_target_line:
        for line in bounding_line:
            bounding_a = (line[0][0], line[0][1])
            bounding_b = (line[0][2], line[0][3])

            line_to_check_a = line_target_line.get(key)[0]
            line_to_check_b = line_target_line.get(key)[1]

            line_bounding = LineString([bounding_a, bounding_b])
            line_to_check = LineString([line_to_check_a, line_to_check_b])

            intersection_point = line_bounding.intersection(line_to_check)

            if not intersection_point.is_empty:
                # if intersection_point.xy[1][0] > 340:
                    print(intersection_point)
                    # return intersection_point.xy[0]
                    result[key] = intersection_point
    return result

img = cv2.imread("./data/video/doroga_s_1_init.png")
plt.imshow(img)
# detect_lines("./data/video/doroga_s_1_init.png", None)