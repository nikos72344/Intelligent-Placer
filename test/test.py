import sys
import lib.PolyDetector as pd
import logging

poly_detector = pd.PolyDetector()

logging.basicConfig(level=logging.DEBUG)

poly_detector.set_logger(logging.getLogger('poly_detector'))

if poly_detector.detect('../1.jpg'):
    poly_detector.save_result('output_1.jpg')
else:
    print('Something wrong', file=sys.stderr)
