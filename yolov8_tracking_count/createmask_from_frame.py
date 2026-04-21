import cv2
import numpy as np
def create_mask_from_first_frame(video_path, mask_path):
    """
    从视频的第一帧中创建掩膜
    :param video_path: 视频文件路径
    :param mask_path: 掩膜保存路径
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    if not success:
        print("无法读取视频的第一帧")
        return

    # 显示第一帧图像，让用户选择四个点
    points = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", img)
            key = cv2.waitKey(0)
            if len(points) < 3:
                print("选取点集过少")
            if key == 13 or 10:
                cv2.polylines(img, [np.array(points)], True, (0, 255, 0), 2)
                cv2.imshow("Select Points", img)
                # 保存掩膜
                mask = np.zeros_like(img)
                cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
                cv2.imwrite(mask_path, mask)
                print("掩膜已保存到", mask_path)
                cap.release()
                cv2.destroyAllWindows()

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", on_EVENT_LBUTTONDOWN)
    cv2.imshow("Select Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()