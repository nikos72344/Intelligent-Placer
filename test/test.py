import sys
import lib.PolyDetector as pd
import lib.ObjectDetector as obd
import logging
import cv2

poly_detector = pd.PolyDetector()
object_detector = obd.ObjectDetector()

logging.basicConfig(level=logging.DEBUG)

poly_detector.set_logger(logging.getLogger('poly_detector'))
object_detector.set_logger(logging.getLogger('object_detector'))

path = '/Users/mikita/Documents/GitHub/Intelligent-Placer/12_1.jpg'
output_name = 'output_2'

if poly_detector.detect(path):
    poly_detector.save_result(output_name + '_poly_paper' + '.jpg')
    if not object_detector.detect(path, poly_detector):
        exit(1)

    object_detector.save_result(output_name + '_obj' + '.jpg')

    contours = [poly_detector.get_poly_contour()]
    contours.append(poly_detector.get_paper_contour())

    objects = object_detector.get_object_contours()
    for obj in objects:
        contours.append(obj)

    file = cv2.imread(path)
    cv2.drawContours(file, contours, -1, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.imwrite(output_name + '.jpg', file)

else:
    print('Something wrong', file=sys.stderr)
