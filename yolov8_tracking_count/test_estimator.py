# 导入所需
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_class_color, get_stationary_rois,process_roi
from createmask_from_frame import create_mask_from_first_frame
from getLine import select_two_points_from_first_frame
import threading

# 检查 GPU 是否可用
import torch
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复导入 OpenCV 库的问题
video_path = "./Videos/traffic.mp4"
mask_path = './static/mask.png'
create_mask_from_first_frame(video_path,mask_path)
#选择视频源
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
#获取视频流的帧率
_,_,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))

# 加载 YOLO 模型，并确保模型在 GPU 上运行
model = YOLO("yolo11n.pt") 
#model = YOLO("runs/detect/t6/best.pt")  
if torch.cuda.is_available():
    model.to("cuda")

# 读取静态图像
mask = cv2.imread("static/mask.png")
# mainCounter = cv2.resize(mainCounter, (700, 250))
outCounter = cv2.imread("static/out_count.png", cv2.IMREAD_UNCHANGED)
if outCounter.shape[2] == 3:  # 如果是 RGB
    # 添加全不透明 alpha 通道
    alpha_channel = 255 * np.ones((outCounter.shape[0], outCounter.shape[1], 1), dtype=outCounter.dtype)
    outCounter = np.concatenate([outCounter, alpha_channel], axis=2)

# 初始化 DeepSort 跟踪器
tracker = DeepSort(
    max_iou_distance=0.5,
    max_age=5,
    n_init=5,
    nms_max_overlap=1.8,
    max_cosine_distance=0.32,
    )

#区域限制
limitsUp = select_two_points_from_first_frame(video_path)

if len(limitsUp) == 2 and len(limitsUp[0]) == 2 and len(limitsUp[1]) == 2:
    limitsUp = [limitsUp[0][0], limitsUp[0][1], limitsUp[1][0], limitsUp[1][1]]
else:
    print("未成功选择两个点，使用默认值")

# 初始化计数器和字典
totalCountUp = []

clsCounterUp = {'car': 0, 'truck': 0, 'motorcycle': 0, 'bus': 0}
interested_classes = ['car', 'truck', 'motorcycle', 'bus']
PIXELS_PER_METER = 15  # 需要校准：1米对应多少像素
prev_positions = {}  # 存储上一帧位置 {track_id: (x, y)}
speed_history = {}  # 存储速度历史 {track_id: speed}
BASE_FPS = 30               # 基准帧率
STATIONARY_THRESHOLD = 1.5    # 静止速度阈值(km/h)
CONSECUTIVE_FRAMES = 10       # 持续判定帧数
MIN_ACTIVE_DISPLACEMENT = 5   # 有效移动最小位移(像素)
frame_count = 0
stationary_records = {}  # {track_id: (静止帧数, 最后有效速度)}
object_sizes = {}  # 存储各目标的尺寸 {track_id: (width, height)}
stationary_objects = set()
BASE_DETECTION_INTERVAL = 5  # 全图检测间隔帧数
stationary_last_detected = {}  # {track_id: 最后检测的帧数}
detect_args = {
    'conf': 0.65,  # 置信度阈值
    'iou': 0.45,  # IOU阈值
    'imgsz': 480,  # 输入尺寸
    'half': True,  # 启用半精度推理
    'device': '0' if torch.cuda.is_available() else 'cpu',  # 指定设备
    'stream': True,  # 保持流式处理
    'verbose': False  # 关闭冗余输出
}
while True:
    frame_count += 1  # 每帧递增
    # 读取视频帧
    actual_fps = fps if fps > 0 else BASE_FPS  # 防止无效帧率
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)
    # 在图像上叠加静态图像
    img = cvzone.overlayPNG(img, outCounter, (0, 0))


    if frame_count % BASE_DETECTION_INTERVAL == 0 or not stationary_objects:
        # 全量检测模式
        results = model(imgRegion,  **detect_args)
    else:
        # ROI检测模式
        rois = get_stationary_rois(
            stationary_objects,
            prev_positions,
            img_shape=img.shape,
            default_size=150
        )

        if rois:
            results = []
            threads = []

            for roi, track_id in rois:  # rois现在包含track_id
                thread = threading.Thread(target=process_roi,
                                          args=(img, roi, track_id, model, results, stationary_last_detected, frame_count))
                threads.append(thread)
                thread.start()
        #         # 如果该静止车辆超过30帧未被检测，则加入检测列表
        #         if frame_count - stationary_last_detected.get(track_id, 0) > 60:
        #             x1, y1, x2, y2 = roi
        #             roi_img = img[y1:y2, x1:x2]
        #             results.extend(model(roi_img, stream=True, conf=0.7))
        #             stationary_last_detected[track_id] = frame_count  # 更新最后检测帧
        else:
            results = []

    detections = list()

    # 处理检测结果，r是该帧图像的所有检测目标，box是所有检测目标里的一个目标（按坐标来排序）
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = model.names[cls]
            if currentClass in interested_classes and conf > 0.7:
                detections.append(([x1, y1, w, h], conf, cls))


    # 画上下限制线
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), thickness=5)

    # 使用 DeepSort 进行目标跟踪
    tracks = tracker.update_tracks(detections, frame=img)


    # 处理跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        w, h = x2 - x1, y2 - y1
        co_ord = [x1, y1]
        cx, cy = x1 + w / 2.0, y1 + h / 2.0 #高精度浮点坐标运算
        cx1, cy1 = x1 + w // 2, y1 + h // 2
        # 记录或更新目标尺寸
        object_sizes[track_id] = (w, h)
        #定期清除不再跟踪的车辆
        if frame_count % 200 == 0:
            active_ids = {t.track_id for t in tracks if t.is_confirmed()}
            for tid in list(prev_positions.keys()):
                if tid not in active_ids:
                    del prev_positions[tid]
                    del speed_history[tid]
                    del stationary_records[tid]
        # 初始化记录
        if track_id not in prev_positions:
            prev_positions[track_id] = (cx, cy, frame_count)
            speed_history[track_id] = 0
            stationary_records[track_id] = (0, 0)
            continue

        # 计算位移和时间差
        prev_x, prev_y, prev_frame = prev_positions[track_id]
        frames_passed = frame_count - prev_frame
        dx, dy = cx - prev_x, cy - prev_y
        pixel_distance = math.sqrt(dx * dx + dy * dy)

        # 静止状态检测
        if pixel_distance < MIN_ACTIVE_DISPLACEMENT:
            stationary_frames, last_valid_speed = stationary_records[track_id]
            stationary_frames += 1

            if stationary_frames >= CONSECUTIVE_FRAMES:
                # 确认为静止状态
                current_speed = 0
                stationary_records[track_id] = (CONSECUTIVE_FRAMES, 0)
            else:
                # 过渡状态：逐渐衰减速度
                current_speed = last_valid_speed * (1 - stationary_frames / CONSECUTIVE_FRAMES)
                stationary_records[track_id] = (stationary_frames, last_valid_speed)
        else:
            # 有效移动状态
            dynamic_ppm = PIXELS_PER_METER * (1 + (cy / img.shape[0]) * 0.3)
            current_speed = (pixel_distance * actual_fps * 3.6) / (dynamic_ppm * frames_passed)
            current_speed = max(0, min(120, current_speed))  # 限制合理范围

            # 更新静止记录
            if current_speed > STATIONARY_THRESHOLD:
                stationary_records[track_id] = (0, current_speed)
            else:
                stationary_records[track_id] = (CONSECUTIVE_FRAMES, 0)

        # 更新位置历史
        prev_positions[track_id] = (cx, cy, frame_count)
        speed_history[track_id] = current_speed

        # 显示处理（静止车辆显示0）
        display_speed = int(round(current_speed)) if current_speed >= 1 else 0

         # 获取对象类别和颜色
        cls = track.get_det_class()
        currentClass = model.names[cls]
        clsColor = get_class_color(currentClass)

        # 在帧上绘制跟踪信息
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=clsColor)
        cvzone.putTextRect(
            img,
            text=f"{model.names[cls]} {display_speed} km/h",
            pos=(max(0, x1), max(35, y1)),
            colorR=clsColor,
            scale=1,
            thickness=1,
            offset=2)


        #在帧上绘制速度信息
        cv2.circle(img, (cx1, cy1), radius=5, color=clsColor, thickness=cv2.FILLED)

        # 处理上下计数
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(track_id) == 0:
                totalCountUp.append(track_id)
                clsCounterUp[currentClass] += 1
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255, 255, 255), thickness=3)


    # 在帧上显示统计信息
    cv2.putText(img, str(len(totalCountUp)), (565, 112), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, str(clsCounterUp["car"]), (50, 32), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, str(clsCounterUp["truck"]), (50, 66), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, str(clsCounterUp["motorcycle"]), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, str(clsCounterUp["bus"]), (50, 131), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0), 2)

    # 在窗口中显示帧
    cv2.imshow('Traffic Monitoring', img)
    # 等待键盘输入
    cv2.waitKey(1)
