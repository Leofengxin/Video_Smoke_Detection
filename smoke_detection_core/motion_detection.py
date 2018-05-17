import logging
import cv2

def img_to_block(rows, cols, block_size):
    result = []
    for c in range(0, cols - block_size, block_size):
        for r in range(0, rows - block_size, block_size):
            box = (r, c, block_size, block_size)
            result.append(box)
    return result

def motion_detection(frames, location_list, block_size):
    gray_cur = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
    # Calculate integrogram

    # diff = cv2.absdiff(gray_background, gray_cur)
    # int_diff = cv2.integral(diff)
    # # This is a key parameter. Change this value can control motion_block number.
    # threshold = block_size * block_size
    # result = list()
    # for pt in iter(location_list):
    #     xx, yy, _bz, _bz = pt
    #     t11 = int_diff[xx, yy]
    #     t22 = int_diff[xx + block_size, yy + block_size]
    #     t12 = int_diff[xx, yy + block_size]
    #     t21 = int_diff[xx + block_size, yy]
    #     block_diff = t11 + t22 - t12 - t21
    #     if block_diff > threshold:
    #         result.append((xx, yy, block_size, block_size))
    # return result

    # T = block_size * block_size / 2
    # # T = 30
    # TH = 4
    # result = list()
    #
    # diff = cv2.absdiff(gray_background, gray_cur)
    # diff /= TH
    #
    # int_diff = cv2.integral(diff, -1)
    # idx = 0
    # for pt in iter(location_list):
    #     xx, yy, d1, d2 = pt
    #     t11 = int_diff[xx, yy]
    #     t22 = int_diff[xx + block_size, yy + block_size]
    #     t12 = int_diff[xx, yy + block_size]
    #     t21 = int_diff[xx + block_size, yy]
    #     block_diff = t11 + t22 - t12 - t21
    #     if block_diff > T:
    #         result.append(idx)
    #     idx += 1
    #     return result



    # TH = 8
    # diff = cv2.absdiff(gray_background, gray_cur)
    # diff = diff / TH
    # int_diff = cv2.integral(diff)
    # # This is a key parameter. Change this value can control motion_block number.
    # threshold = block_size * block_size / 2
    # # threshold = 400
    # result = list()
    # for pt in iter(location_list):
    #     xx, yy, _bz, _bz = pt
    #     t11 = int_diff[xx, yy]
    #     t22 = int_diff[xx + block_size, yy + block_size]
    #     t12 = int_diff[xx, yy + block_size]
    #     t21 = int_diff[xx + block_size, yy]
    #     block_diff = t11 + t22 - t12 - t21
    #     if block_diff > threshold:
    #         result.append((xx, yy, block_size, block_size))
    # return result


    history = len(frames)    # 训练帧数
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)
    i = 0
    num = 0-history
    while i < history:
        fg_mask = bs.apply(frames[num])  # 获取 foreground mask
        num += 1
        i += 1

    th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

    cv2.imshow("motion_detection", dilated)

    int_diff = cv2.integral(dilated)

    result = list()
    for pt in iter(location_list):
        xx, yy, _bz, _bz = pt
        t11 = int_diff[xx, yy]
        t22 = int_diff[xx + block_size, yy + block_size]
        t12 = int_diff[xx, yy + block_size]
        t21 = int_diff[xx + block_size, yy]
        block_diff = t11 + t22 - t12 - t21
        if block_diff > 0:
            result.append((xx, yy, block_size, block_size))
    return result


# TODO: This method need to rewrite.
def motion_detection_with_optical_flow(self,img, gray, back_gray):
    pass

class motion_detector_factory(object):
    detectors = {}
    detectors['background_substraction'] = motion_detection

    def get_motion_detector(self, detector_name):
        if detector_name in self.detectors:
            motion_detector = self.detectors[detector_name]
            logging.info('Motion detector:{}.'.format(detector_name))
            return motion_detector
        else:
            logging.error('Motion detector:{} is not supported, please check your code!'.format(detector_name))