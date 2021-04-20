import os
import textwrap
from typing import Dict, List, Optional, Tuple
from PIL import Image

import imageio
import numpy as np
import tqdm


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):

    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )

    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


frames = []
for i in range(1, 10000):
    frame = Image.open(f'/nethome/qluo49/iGibsonChallenge2021/pictures/{i}')
    frames.append(frame)
images_to_video(frames, '/nethome/qluo49/iGibsonChallenge2021/videos', 'social_nav')


