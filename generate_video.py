# import os
# import textwrap
# from typing import Dict, List, Optional, Tuple
# from PIL import Image

# import imageio
# import numpy as np
# import tqdm

# def images_to_video(
#     images: List[np.ndarray],
#     output_dir: str,
#     video_name: str,
#     fps: int = 10,
#     quality: Optional[float] = 5,
#     **kwargs,
# ):

#     assert 0 <= quality <= 10
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
#     writer = imageio.get_writer(
#         os.path.join(output_dir, video_name),
#         fps=fps,
#         quality=quality,
#         **kwargs,
#     )

#     for im in tqdm.tqdm(images):
#         writer.append_data(im)
#     writer.close()


# frames = []
# for i in range(1, 10000):
#     frame = Image.open(f'/nethome/qluo49/iGibsonChallenge2021/pictures/{i}.png')
#     frames.append(frame)
# images_to_video(frames, '/nethome/qluo49/iGibsonChallenge2021/videos', 'social_nav')

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image

def Pic2Video():
    imgPath = f'/nethome/qluo49/iGibsonChallenge2021/pictures/' 
    videoPath = '/nethome/qluo49/iGibsonChallenge2021/videos/social_nav.avi'  
 
    images = os.listdir(imgPath)
    fps = 15  
 
    fourcc = VideoWriter_fourcc(*"XVID")
 
    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  
        print(im_name)
        videoWriter.write(frame)

    videoWriter.release()
    cv2.destroyAllWindows()


def Video2Pic():
    videoPath = "youvideoPath"  # 读取视频路径
    imgPath = "youimgPath"  # 保存图片路径
 
    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        cv2.imwrite(imgPath + str(frame_count).zfill(4), frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")


if __name__ == '__main__':
    # Video2Pic()
    Pic2Video()




