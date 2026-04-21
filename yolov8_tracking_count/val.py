from ultralytics import YOLO
# 加载训练好的模型
model = YOLO("runs/detect/t6/best.pt")
# 进行模型验证
results = model.val(data="yolo_tf.yaml",batch=16)