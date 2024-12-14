import cv2
import numpy as np

def region_of_interest(img, vertices):
    # 创建一个与原始图像相同大小的全黑掩码图像
    mask = np.zeros_like(img)
    # 在掩码图像上根据给定的顶点坐标绘制填充的多边形区域
    cv2.fillPoly(mask, vertices, 255)
    # 将原始图像与掩码进行按位与操作，得到感兴趣区域内的图像部分
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def fit_lane_lines(lines, height, width):
    left_points = []
    right_points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线斜率
            slope = (y2 - y1) / (x2 - x1)
            if 0.5 < abs(slope) < 2.0:
                if slope < 0 and x1 < width / 2 and x2 < width / 2:
                    # 左车道线
                    left_points.extend([(x1, y1), (x2, y2)])
                elif slope > 0 and x1 > width / 2 and x2 > width / 2:
                    # 右车道线
                    right_points.extend([(x1, y1), (x2, y2)])
    left_lane = None
    right_lane = None
    if left_points and right_points:
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        # 用多项式拟合车道线
        left_lane = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        right_lane = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
    return left_lane, right_lane

def get_lane_points(lane, height, width):
    if lane is None:
        return None
    # 生成一系列用于绘制车道线的坐标点
    plot_y = np.linspace(height * 0.6, height - 1, height // 3)
    fit_x = lane[0] * plot_y ** 2 + lane[1] * plot_y + lane[2]
    # 确保车道线在图像范围内
    fit_x = np.clip(fit_x, 0, width - 1)
    points = np.column_stack((fit_x, plot_y)).astype(np.int32)
    return points

def detect_lane(frame):
    # 将图像转换为HLS颜色空间
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # 设定白色阈值
    lower_white = np.array([0, 138, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    white_image = cv2.bitwise_and(frame, frame, mask=white_mask)
    gray = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
    # 定义感兴趣区域的顶点
    height, width = frame.shape[:2]
    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    # 对感兴趣区域进行掩码处理
    masked_image = region_of_interest(gray, vertices)
    # 对图像进行高斯模糊处理
    blurred = cv2.GaussianBlur(masked_image, (7, 7), 0)
    # 使用Canny算子检测图像边缘
    binary_output = cv2.Canny(blurred, 50, 150)
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(binary_output, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=100)
    # 拟合车道线
    left_lane, right_lane = fit_lane_lines(lines, height, width)
    lane_image = np.zeros_like(frame)
    # 绘制检测到的车道区域
    if left_lane is not None and right_lane is not None:
        left_lane_points = get_lane_points(left_lane, height, width)
        right_lane_points = get_lane_points(right_lane, height, width)
        if left_lane_points is not None and right_lane_points is not None:
            cv2.polylines(lane_image, [left_lane_points], isClosed=False, color=(0, 255, 0), thickness=5)
            cv2.polylines(lane_image, [right_lane_points], isClosed=False, color=(0, 255, 0), thickness=5)
            pts = np.vstack((left_lane_points, right_lane_points[::-1]))
            cv2.fillPoly(lane_image, [pts], color=(255, 0, 0))
    # 将检测到的车道线叠加到原始图像上
    result = cv2.addWeighted(frame, 1, lane_image, 0.5, 0)
    return result

# 打开视频文件
cap = cv2.VideoCapture('sunny_video.mp4')
# 获取视频参数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 创建输出视频对象
out = cv2.VideoWriter('lane_detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        lane_detection = detect_lane(frame)
        out.write(lane_detection)
        cv2.imshow('检测', lane_detection)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()