from collections import Counter

import cv2
import torch
import numpy as np
from scipy.stats import mode
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomCrop, Resize,
                                    ToTensor)

from models import GTARes18Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_video = './sample_video_3.mp4'

video_stream = cv2.VideoCapture(sample_video)
print('Approximate frame count:', video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

transform = Compose([
    RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = GTARes18Net(9).to(device)
model_file = './trained_models/gta_res_source.pt'
model.load_state_dict(torch.load(model_file), strict=False)
model.eval()

dataset = ImageFolder('./t_data')
idx_to_class = dict()
for class_name, idx in dataset.class_to_idx.items():
    idx_to_class[idx] = class_name

class_to_color = {
    'Arrest': (255, 0, 0),
    'Arson': (0, 0, 255),
    'Assault': (106, 9, 125),
    'Explosion': (252, 130, 16),
    'Fight': (0, 161, 171),
    'Normal': (0, 255, 0),
    'Robbery': (68, 39, 39),
    'Shooting': (154, 31, 64),
    'Vandalism': (235, 99, 131),
}

count = 0
frame_labels = list()
while True:
    retval, frame = video_stream.read()

    if not retval:
        break

    image = Image.fromarray(frame)
    ready_image = transform(image).to(device)
    batch_view = ready_image.view(1, 3, 224, 224)
    y_pred = model(batch_view)
    predictions = y_pred.max(1)[1]
    frame_labels.append(predictions[0].item())

    count += 1
    if count % 500 == 0:
        print(count, 'frames processed')

video_label, _ = mode(frame_labels)
print('Crime on video:', idx_to_class[video_label[0]])

# Reopen video stream
video_stream = cv2.VideoCapture(sample_video)
fps = video_stream.get(cv2.CAP_PROP_FPS)
frame_size = (2 * int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
              2 * int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
piechart_bounds = (
    (frame_size[0] - 80, 20),
    (frame_size[0] - 20, 80),
)
output_video = cv2.VideoWriter(
    './output_video_3.avi',
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    int(fps),
    frame_size,
)

frame_number = 0
default_font = ImageFont.truetype(
    '/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf', 24)
while True:
    retval, frame = video_stream.read()
    frame_number += 1

    if not retval:
        break

    current_frames = frame_labels[:frame_number]
    image = Image.fromarray(frame).resize(frame_size)
    draw_obj = ImageDraw.Draw(image)
    video_label, _ = mode(current_frames)
    video_class = idx_to_class[video_label[0]]
    draw_obj.text((20, frame_size[1] - 40),
                  f'Video label: {video_class}'.upper(),
                  font=default_font,
                  fill=class_to_color.get(video_class, (255, 255, 255)))

    counts = Counter(current_frames)
    bottom_padding = 90
    start_angle = 0
    for idx, class_name in idx_to_class.items():
        angle = counts[idx] / len(current_frames) * 360
        next_angle = start_angle + angle
        draw_obj.pieslice(piechart_bounds,
                            start_angle,
                            next_angle,
                            fill=class_to_color[class_name])
        start_angle = next_angle

        if counts[idx] > len(current_frames) / 20:
            text_size = draw_obj.textsize(class_name.upper(), font=default_font)
            t_point = (frame_size[0] - 10 - text_size[0], bottom_padding)
            bottom_padding += text_size[1] + 10

            draw_obj.text(t_point,
                          class_name.upper(),
                          align='right',
                          font=default_font,
                          fill=class_to_color[class_name])

    output_video.write(np.asarray(image))
