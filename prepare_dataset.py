# pylint: disable=invalid-name,missing-docstring
import os
import subprocess
import urllib.parse
import uuid


def read_temporal_data(temporal_path: str):
    temporal_data = dict()
    for line in open(temporal_path, 'r'):
        name, v_cls, s_frm_1, f_frm_1, s_frm_2, f_frm_2 = line.strip().split()
        temporal_data[name] = {
            'class': v_cls,
            'action_1': (int(s_frm_1), int(f_frm_1)),
            'action_2': (int(s_frm_2), int(f_frm_2)),
        }
    return temporal_data


source_path = '/home/daniil/Downloads/UCF-Crime/Videos'
result_path = '/home/daniil/Documents/Projects/University/Thesis/frames2'
if not os.path.exists(result_path):
    os.mkdir(result_path)

normal_frames = os.path.join(result_path, 'Normal')
if not os.path.exists(normal_frames):
    os.mkdir(normal_frames)

temporal_data = read_temporal_data('./temporal_data.txt')
err_log = open('./err.log', 'w')
for video_class in os.listdir(source_path):
    frames_dir = os.path.join(result_path, video_class)
    class_path = os.path.join(source_path, video_class)
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    for idx, video in enumerate(os.listdir(class_path)):
        video_path = os.path.join(class_path, video).strip()

        print(f'Start working on {os.path.abspath(video_path)}')
        video_name, video_ext = os.path.splitext(os.path.basename(video_path))
        video_name = urllib.parse.unquote(video_name)

        command = ('ffmpeg', '-i', os.path.abspath(video_path), '-vf',
                   'select=not(mod(n\\,120))', '-vsync', 'vfr', '-hide_banner',
                   '-threads', '16',
                   os.path.join(frames_dir, f'{idx:03d}-%06d.jpg'))
        try:
            subprocess.check_call(command,
                                  stdout=subprocess.DEVNULL,
                                  stderr=err_log)
        except subprocess.CalledProcessError as exc:
            print(f'ffmpeg failed with return code {exc.returncode}')

        video_frames = sorted([
            frame for frame in os.listdir(frames_dir)
            if frame.startswith(f'{idx:03d}-')
        ])

        if video in temporal_data.keys() and temporal_data[video] != 'Normal':
            frames_data = temporal_data[video]
            frames_to_move = set()
            for action in ['action_1', 'action_2']:
                if frames_data[action][0] == -1:
                    continue

                start_frame, stop_frame = frames_data[action]
                for frm_idx, frame in enumerate(video_frames):
                    frm_num = frm_idx * 120
                    if not start_frame < frm_num < stop_frame:
                        frames_to_move.add(os.path.join(frames_dir, frame))

            for frame_path in frames_to_move:
                uuid_idx = str(uuid.uuid4())
                os.rename(
                    frame_path,
                    os.path.join(result_path, 'Normal',
                                 f'moved-{uuid_idx}.jpg'))

    frames_to_rename = list(os.listdir(frames_dir))
    for frame in frames_to_rename:
        frame_idx = frame.split('-')[0]
        frame_path = os.path.join(frames_dir, frame)
        uuid_idx = str(uuid.uuid4())
        os.rename(frame_path,
                  os.path.join(frames_dir, f'{frame_idx}-{uuid_idx}.jpg'))
