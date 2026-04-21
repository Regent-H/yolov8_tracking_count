import cv2
import os

def process_video(video_path, mask_path, save_dir, save_step):

    """
    处理视频并保存ROI区域的图像
    :param video_path: 视频文件路径
    :param mask_path: 掩膜保存路径
    :param save_dir: 保存图像的目录
    :param save_step: 保存图像的步长
    """
    # 创建掩膜
    mask_path = './static/mask.png'

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    num = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(1280, 720))
        num += 1
        if num % save_step == 20:
            # 读取掩膜
            mask = cv2.imread(mask_path)
            # 应用掩膜
            masked_frame = cv2.bitwise_and(frame, mask)
            # 保存图像
            save_path = os.path.join(save_dir, f'{num}.jpg')
            cv2.imwrite(save_path, masked_frame)
            print(f"Saved {save_path}")

    cap.release()
    cv2.destroyAllWindows()

# 定义参数
video_path = "./Videos/traffic.mp4"
mask_path = './static/mask.png'
save_dir = './demo2/'
save_step = 50

# 处理视频
process_video(video_path, mask_path, save_dir, save_step)