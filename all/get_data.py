# 获取源图片
import cv2
def getPic(video_path, num):
    video = cv2.VideoCapture(video_path)
    frame_num = 0  # 读取的帧数
    pic_num = 1  # 保存图片的数目
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            frame_num += 1
            # 每间隔100帧提取一张图片
            if frame_num % 100 == 0 and pic_num < num+1:
                output_name = "F:/OpenCV/source/%d.jpg" % pic_num
                pic_num += 1
                cv2.imwrite(output_name, frame)
        else:
            break
    video.release()
if __name__ == "__main__":
    path = 'F:/OpenCV/video.mp4'   # 视频路径
    num = 600     # 水果源图片总数
    getPic(path, num)
    print("Done!")