import os
import cv2


class PolyDetector:
    def __init__(self):
        self.__path = ''
        self.__logger = None
        self.__file_orig = None

        self.__contours = None
        self.__paper_vertex = None
        self.__poly_vertex = None

    def set_logger(self, logger):
        self.__logger = logger

    def __read_file_orig(self, path):
        if not os.path.exists(path):
            if self.__logger is not None:
                self.__logger.error('File \'%s\' doesn\'t exist' % path)
            return False

        self.__path = path
        self.__file_orig = cv2.imread(self.__path)

        if self.__logger is not None:
            self.__logger.info('Original file read successfully')

        return True

    def __find_contours(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.__logger is not None:
            self.__logger.info('Grayscale file read successfully')

        img = cv2.blur(img, (10, 15))
        _, img = cv2.threshold(img, 10, 280, cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.__logger is not None:
            self.__logger.info('Detected %s contours in file', len(contours))

        return contours

    def __filter_contours(self, contours):
        final_contours = []

        for cnt in contours:
            moment = cv2.moments(cnt)

            if moment['m00'] == 0:
                if self.__logger is not None:
                    self.__logger.debug('Excluded contour with moment m00 = 0')
                continue

            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            if self.__logger is not None:
                self.__logger.debug('(x = %s; y = %s)' % (cx, cy))

            if cy < self.__file_orig.shape[0] / 2:
                final_contours.append(cnt)
            elif self.__logger is not None:
                self.__logger.info('Contour with the center (%s; %s) was excluded' % (cx, cy))

        if self.__logger is not None:
            self.__logger.info('%s contours left', len(final_contours))

        return final_contours

    def __find_vertex(self, contours):
        vertex = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    vertex.append((n[i], n[i + 1]))
                    if self.__logger is not None:
                        self.__logger.info('Found vertex: (%s; %s)' % (n[i], n[i + 1]))
                i = i + 1

        if self.__logger is not None:
            self.__logger.info('Found %s vertex in paper area' % len(vertex))

        return vertex

    def __determine_vertex(self, vertex):
        x_sorted = sorted(vertex, key=lambda tup: tup[0])

        if len(x_sorted) < 4:
            if self.__logger is not None:
                self.__logger.error('Amount of detected vertex is less than 4: check your image')
            return False

        temp = [x_sorted[0], x_sorted[1], x_sorted[-2], x_sorted[-1]]
        self.__paper_vertex = sorted(temp, key=lambda tup: tup[1])

        if self.__logger is not None:
            self.__logger.info('Found paper vertex')
            self.__logger.debug('Paper vertex are: %s' % self.__paper_vertex)

        x_sorted.pop(0)
        x_sorted.pop(0)
        x_sorted.pop(-1)
        x_sorted.pop(-1)

        if len(x_sorted) < 3 or len(x_sorted) > 6:
            if self.__logger is not None:
                self.__logger.error('Incorrect polygon vertex amount: %s' % len(x_sorted))
            return False

        self.__paper_vertex = sorted(x_sorted, key=lambda tup: tup[1])

        if self.__logger is not None:
            self.__logger.info('Found polygon vertex')
            self.__logger.debug('Polygon vertex are: %s' % self.__paper_vertex)

        return True

    def detect(self, path):
        if not self.__read_file_orig(path):
            return False

        contours = self.__filter_contours(self.__find_contours(path))

        if len(contours) == 1:
            if self.__logger is not None:
                self.__logger.error('Detected only potential paper contour: couldn\'t find polygon')
            return False

        if len(contours) == 2:
            if self.__logger is not None:
                self.__logger.error('Detected only two contours in paper area: check your image')
            return False

        self.__contours = contours
        cv2.drawContours(self.__file_orig, self.__contours, -1, (0, 0, 255), 5, cv2.LINE_AA)

        result = self.__determine_vertex(self.__find_vertex(contours))

        cv2.circle(self.__file_orig, self.__paper_vertex, radius=3, color=(255, 0, 0), thickness=-1)
        cv2.circle(self.__file_orig, self.__poly_vertex, radius=3, color=(0, 255, 0), thickness=-1)

        return result

    def get_paper_vertex(self):
        return self.__paper_vertex

    def get_poly_vertex(self):
        return self.__poly_vertex

    def save_result(self, path='./output/output.jpg'):
        cv2.imwrite(path, self.__file_orig)
