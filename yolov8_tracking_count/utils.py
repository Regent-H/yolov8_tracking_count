import math

from sympy.categories.baseclasses import Class

# 定义颜色调色板
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def get_class_color(cls):
    """
    获取对象类别对应的颜色

    Args:
    - cls (str): 对象的类别

    Returns:
    - tuple: RGB颜色值
    """
    if cls == 'car':
        color = (204, 51, 0)
    elif cls == 'truck':
        color = (22, 82, 17)
    elif cls == 'motorbike':
        color = (255, 0, 85)
    elif cls == 'bus':
        color = (33, 255, 85)
    else:
        # 对于未知类别，使用调色板生成动态颜色
        color = [int((p * (2 ** 2 - 14 + 1)) % 255) for p in palette]
    return tuple(color)


def estimatedSpeed(location1, location2):
    """
    估算目标速度
    Args:
    - location1 (list): 起始位置坐标 [x1, y1]
    - location2 (list): 结束位置坐标 [x2, y2]
    Returns:
    - int: 估算速度（千米/小时）
    """
    # 计算两点之间的欧几里得距离（像素）
    d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))

    # 设置每米的像素数（像素每米）
    ppm = 2.5  # 可以根据对象离摄像头的距离动态调整这个值

    # 将像素距离转换为实际距离（米）
    d_meters = d_pixel / ppm

    # 时间常数，用于速度估算
    time_constant = 15 * 3.6  # 这个值可以根据实际情况调整

    # 估算速度（千米/小时）
    speed = (d_meters * time_constant) / 100

    return int(speed)

object_sizes = {}
def get_stationary_rois(stationary_objects, prev_positions, img_shape, default_size=100):
    """生成静止目标周围的检测区域

    参数:
        stationary_objects: 静止目标ID集合
        prev_positions: 位置记录字典 {track_id: (x, y, frame)}
        img_shape: 图像形状 (height, width)
        default_size: 默认检测区域边长

    返回:
        list: 每个ROI的坐标[x1,y1,x2,y2]
    """
    rois = []
    img_h, img_w = img_shape[:2]  # 获取图像高度和宽度
    for tid in stationary_objects:
        if tid in prev_positions:
            x, y, _ = prev_positions[tid]

            # 确定检测区域大小
            if tid in object_sizes:  # 如果知道目标尺寸
                w, h = object_sizes[tid]
                size = int(max(w, h) * 1.5 ) # 使用目标尺寸的1.5倍
            else:
                size = default_size  # 默认大小

                # 计算ROI边界（确保不超出图像范围）
                half_size = size // 2
                x1 = max(0, int(x - half_size))
                y1 = max(0, int(y - half_size))
                x2 = min(img_w, int(x + half_size))
                y2 = min(img_h, int(y + half_size))

                # 只添加有效的ROI区域
                if x2 > x1 and y2 > y1:
                    rois.append([x1, y1, x2, y2])

    return rois

# utils.py
def process_roi(img, roi, track_id, model, results, stationary_last_detected, frame_count):
    if frame_count - stationary_last_detected.get(track_id, 0) > 100:
        x1, y1, x2, y2 = roi
        roi_img = img[y1:y2, x1:x2]
        roi_results = model(roi_img, stream=True, conf=0.7)
        results.extend(roi_results)
        stationary_last_detected[track_id] = frame_count  # 更新最后检测帧

