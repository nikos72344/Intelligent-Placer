import os
import logging
import cv2


class PolyDetector:
    def __init__(self):
        self.__path = ''
        self.__logger = None

        self.__paper_contour = None
        self.__paper_vertex = None

        self.__poly_contour = None
        self.__poly_vertex = None

    def set_logger(self, logger: logging):
        self.__logger = logger

    def __is_file_exist(self, path):
        if not os.path.exists(path):
            if self.__logger is not None:
                self.__logger.error('File \'%s\' doesn\'t exist' % path)
            return False

        self.__path = path

        return True

    # Находим контуры на изображении
    def __find_contours(self, path):
        # Считываем изображение в серых тонах
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.__logger is not None:
            self.__logger.info('Grayscale file read successfully')

        # Сглаживаем прочитанное изображение
        img = cv2.blur(img, (11, 11))
        # Проводим бинаризацию
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

        # Находим контуры на преобразованном изображении
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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

            # Для каждого контура находим его центр
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            if self.__logger is not None:
                self.__logger.debug('Found contour center (x = %s; y = %s)' % (cx, cy))

            # Допускаем только контура, центр которых находится в верхней половине изображения
            if cy < 4000 / 2:
                final_contours.append(cnt)
            elif self.__logger is not None:
                self.__logger.debug('Contour with the center (%s; %s) was excluded' % (cx, cy))

        if self.__logger is not None:
            self.__logger.info('%s contours left', len(final_contours))

        return final_contours

    def __get_poly_paper_contours(self, contours):
        contour_areas = []

        # Находим периметр для каждого контура в верхней части изображения
        for cnt in contours:
            contour_areas.append((cv2.arcLength(cnt, True), cnt))

        # Сортируем по возрастанию периметры
        contour_areas_sorted = sorted(contour_areas, key=lambda tup: tup[0])

        # Предпоследний - внутренний контур листа, 4-ий с конца - внутренний контур многоугольника
        return contour_areas_sorted[-4][1], contour_areas_sorted[-2][1]

    # Находим вершины распознанных контуров
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
                        self.__logger.debug('Found vertex: (%s; %s)' % (n[i], n[i + 1]))
                i = i + 1

        if self.__logger is not None:
            self.__logger.info('Found %s vertex in paper area' % len(vertex))

        return vertex

    # Определяем, где вершины листа, а где многоугольника
    def __determine_vertex(self, vertex):
        x_sorted = sorted(vertex, key=lambda tup: tup[0])

        if len(x_sorted) < 4:
            if self.__logger is not None:
                self.__logger.error('Amount of detected vertex is less than 4: check your image')
            return False

        # Точки на периферии - вершины листа. Их всегда 4
        temp = [x_sorted[0], x_sorted[1], x_sorted[-2], x_sorted[-1]]
        self.__paper_vertex = sorted(temp, key=lambda tup: tup[1])

        if self.__logger is not None:
            self.__logger.info('Found paper vertex')
            self.__logger.debug('Paper vertex are: %s' % self.__paper_vertex)

        x_sorted.pop(0)
        x_sorted.pop(0)
        x_sorted.pop(-1)
        x_sorted.pop(-1)

        if len(x_sorted) < 3:
            if self.__logger is not None:
                self.__logger.error('Incorrect polygon vertex amount: %s' % len(x_sorted))
            return False

        # Остальные точки - вершины многоугольника
        self.__poly_vertex = sorted(x_sorted, key=lambda tup: tup[1])

        if self.__logger is not None:
            self.__logger.info('Found polygon vertex')
            self.__logger.debug('Polygon vertex are: %s' % self.__poly_vertex)

        return True

    def detect(self, path: str):
        if not self.__is_file_exist(path):
            return False

        contours = self.__filter_contours(self.__find_contours(path))

        # Проверка если программа обнаружила неправильное для задачи количество контуров
        if len(contours) < 4:
            if self.__logger is not None:
                self.__logger.error('Detected only %s contours in paper area: check your image' % len(contours))
            return False

        self.__poly_contour, self.__paper_contour = self.__get_poly_paper_contours(contours)

        if self.__logger is not None:
            self.__logger.info('Detected polygon and paper contours')

        return self.__determine_vertex(self.__find_vertex([self.__paper_contour, self.__poly_contour]))

    def get_paper_contour(self):
        return self.__paper_contour

    def get_paper_vertex(self):
        return self.__paper_vertex

    def get_poly_contour(self):
        return self.__poly_contour

    def get_poly_vertex(self):
        return self.__poly_vertex

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

        if self.__paper_contour is not None:
            cv2.drawContours(file , self.__paper_contour, -1, (0, 0, 255), 5, cv2.LINE_AA)

        if self.__poly_contour is not None:
            cv2.drawContours(file, self.__poly_contour, -1, (255, 0, 0), 5, cv2.LINE_AA)

        if self.__logger is not None:
            self.__logger.debug('Detected contours are drawn')

        if self.__paper_vertex:
            for dot in self.__paper_vertex:
                cv2.circle(file, dot, radius=15, color=(255, 0, 0), thickness=-1)

        if self.__poly_vertex:
            for dot in self.__poly_vertex:
                cv2.circle(file, dot, radius=15, color=(0, 0, 255), thickness=-1)

        if self.__logger is not None:
            self.__logger.debug('Found vertex are drawn')

        cv2.imwrite(path, file)
        if self.__logger is not None:
            self.__logger.info('Result have been saved to \'%s\'' % path)
