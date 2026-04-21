import cv2
def select_two_points_from_first_frame(video_path):
    """
    从视频的第一帧中选择两个点并输出坐标
    :param video_path: 视频文件路径
    :return: 两个点的坐标列表 [(x1, y1), (x2, y2)]
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    if not success:
        print("无法读取视频的第一帧")
        return []

    # 显示第一帧图像，让用户选择两个点
    points = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", img)
            if len(points) == 2:
                print("选中的两个点的坐标为：", points)
                cap.release()
                cv2.destroyAllWindows()

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", on_EVENT_LBUTTONDOWN)
    cv2.imshow("Select Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

if __name__ == '__main__':
    video_path = "./Videos/traffic.mp4"
    print(select_two_points_from_first_frame(video_path))

