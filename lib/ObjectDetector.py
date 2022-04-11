import lib.PolyDetector as pd
import numpy as np
import logging
import cv2
import os


class ObjectDetector:
    def __init__(self):
        self.__path = ''
        self.__logger = None

        self.__objects = []

    def set_logger(self, logger: logging):
        self.__logger = logger

    def __is_file_exist(self, path):
        if not os.path.exists(path):
            if self.__logger is not None:
                self.__logger.error('File \'%s\' doesn\'t exist' % path)
            return False

        self.__path = path

        return True

    def __do_threshold(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        height, width = img.shape
        if height != 4000 or width != 3000:
            if self.__logger is not None:
                self.__logger.error('Wrong file resolution: image must be 3000 x 4000')
            return False
        elif self.__logger is not None:
            self.__logger.info('Grayscale file read successfully')

        img = cv2.medianBlur(img, 51)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2)

        return img

    def __get_vertical_borders(self, horizontal_indices):
        borders = []

        left_border = horizontal_indices[0]
        for i in range(1, len(horizontal_indices)):
            if horizontal_indices[i - 1] != horizontal_indices[i] - 1 and horizontal_indices[i] - horizontal_indices[i - 1] >= 100:
                borders.append((left_border, horizontal_indices[i - 1]))

                if self.__logger is not None:
                    self.__logger.debug('Found  %s x-borders', (left_border, horizontal_indices[i - 1]))

                left_border = horizontal_indices[i]

        borders.append((left_border, horizontal_indices[-1]))

        return borders

    def __get_horizontal_borders(self, vertical_indices):
        borders = []

        left_border = vertical_indices[0]
        for i in range(1, len(vertical_indices)):
            if vertical_indices[i - 1] != vertical_indices[i] - 1 and vertical_indices[i] - vertical_indices[i - 1] >= 100:
                borders.append((left_border, vertical_indices[i - 1]))

                if self.__logger is not None:
                    self.__logger.debug('Found  %s y-borders', (left_border, vertical_indices[i - 1]))

                left_border = vertical_indices[i]

        borders.append((left_border, vertical_indices[-1]))

        return borders

    def __get_object_images(self, img, borders_x, borders_y):
        areas = []

        for point_x in borders_x:
            left = point_x[0]
            right = point_x[1]

            for point_y in borders_y:
                top = point_y[0]
                bottom = point_y[1]

                area = img[top:bottom + 1, left:right + 1]
                areas.append(((left, top), area))

                if self.__logger is not None:
                    self.__logger.debug('Found object area %s', [(left, top), (right, bottom)])

        return areas

    def __detect_objects(self, img):
        horizontal_indices = np.where(np.any(img, axis=0))[0]
        vertical_indices = np.where(np.any(img, axis=1))[0]

        borders_x = self.__get_vertical_borders(horizontal_indices)
        borders_y = self.__get_horizontal_borders(vertical_indices)

        if self.__logger is not None:
            self.__logger.debug('Got %s x-borders' % len(borders_x))
            self.__logger.debug('%s', borders_x)

            self.__logger.debug('Got %s y-borders' % len(borders_y))
            self.__logger.debug('%s', borders_y)

        return self.__get_object_images(img, borders_x, borders_y)

    def __get_objects_concave_hull(self, objects_data, object_part_top):
        result = []

        for data in objects_data:
            contours, _ = cv2.findContours(data[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue

            stack = np.vstack([contours[i] for i in range(len(contours))])
            hull = cv2.convexHull(stack)

            for temp in hull:
                for point in temp:
                    point[0] = point[0] + data[0][0]
                    point[1] = point[1] + data[0][1] + object_part_top

            if cv2.contourArea(hull) >= 500:
                result.append(hull)

        if self.__logger is not None:
            self.__logger.info('Found %s objects' % len(result))

        return result

    def detect(self, path: str, poly_detector: pd.PolyDetector):
        if not self.__is_file_exist(path):
            return False

        img = self.__do_threshold(path)

        object_part_top = poly_detector.get_paper_vertex()[-1][1] + 30
        objects_data = self.__detect_objects(img[object_part_top:, :])
        self.__objects = self.__get_objects_concave_hull(objects_data, object_part_top)

        return True

    def get_object_contours(self):
        return self.__objects

    def save_result(self, path='output.jpg'):
        if not self.__is_file_exist(self.__path):
            return False

        file = cv2.imread(self.__path)
        height, width, channels = file.shape
        if height != 4000 or width != 3000 or channels != 3:
            if self.__logger is not None:
                self.__logger.error('Wrong file resolution: image must be 3000 x 4000')
            return False

        if self.__logger is not None:
            self.__logger.info('Original file read successfully')

        cv2.drawContours(file, self.__objects, -1, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imwrite(path, file)
        if self.__logger is not None:
            self.__logger.info('Result have been saved to \'%s\'' % path)

