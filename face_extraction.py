from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
import os
from os import system
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
import pickle
import subprocess
from multiprocessing import Pool



def draw_features(image, face_landmarks_list, fname=None, invert=False,
                  linewidth=2, facial_features=None):
    """
    Draws facial features on image
    Parameters
    ----------
    image: np.array(:, :, 3)
        RGB
    face_landmarks_list: list(dict(feature:np.array())))
    fname: str or None
        If specified, save to this path
    invert: bool
        Use if input is in GBR
    linewidth: int
    facial_features: list or None
        which features to draw. Default: all of them

    Returns
    -------
    If no filename is given, returns np.array(:, :, 3) image
    """
    if facial_features is None:
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

    if invert:
        image = image[:, :, ::-1]

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:
        for facial_feature in facial_features:
            if np.isfinite(np.all(np.array(face_landmarks[facial_feature]))):
                d.line(face_landmarks[facial_feature], width=linewidth)

    if fname is None:
        return pil_image
    pil_image.format = "PNG"
    pil_image.save(fname)


def detect_faces_video_chunked(video_file, chunks=5):
    converted_fname = video_file[:-4]+'_mpeg4'+video_file[-4:]
    video_file = convert_vcodec(video_file, converted_fname)

    output_paths = split_video_n(video_file, chunks)

    with Pool(chunks) as p:
        results = p.map(detect_faces_video, output_paths)
    return results


def detect_faces_video(video_file, output_video_file=None, fps=None,
                       dimensions=None, output_data_file=None, save_n=1000):
    """
    Detect faces and draw them on video. Return features through time.

    Parameters
    ----------
    video_file: str
    output_file: str or None
    fps: int or None
        default: take from input file
    dimensions: tuple or None
        default: take from input file
    save_n: int or None
        If not None, save every n frames (Default = 1000)

    Returns
    -------
    list(
        dict(
            feature:np.array()
            ),...
        )

    """

    converted_fname = video_file[:-4]+'_mpeg4'+video_file[-4:]
    video_file = convert_vcodec(video_file, converted_fname)

    video_capture = cv2.VideoCapture(video_file)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if dimensions is None:
        dimensions = (int(video_capture.get(3)), int(video_capture.get(4)))

    if output_video_file:
        if fps is None:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('temp.avi', fourcc, fps, dimensions)

    all_features = []
    counter = 0
    for _ in tqdm(range(frame_count)):
        counter += 1
        success, frame = video_capture.read()
        if not success:
            raise Exception('video cannot be read')
        face_landmarks_list = face_recognition.face_landmarks(frame)
        all_features.append(face_landmarks_list)
        if save_n and not (counter % save_n) and output_data_file:
            pickle.dump(all_features, open(
                output_data_file + '_' + str(counter), 'bw'))

        if output_video_file:
            frame2 = draw_features(frame, face_landmarks_list)
            out.write(np.array(frame2))

    if output_video_file:
        out.release()
        """
        * encoding with h264 makes the video about 1/20th the size
        * h264 videos can be played in VLC
        * the `map` arguments take the audio from the original video file
        """
        system('ffmpeg -i temp.avi -i {} -vcodec h264 -map 0:0 -map 1:1 -shortest {}'
               .format(video_file, output_video_file))
        system('rm temp.avi')

    if output_data_file:
        pickle.dump(all_features, open(output_data_file, 'bw'))

    return all_features


def draw_features_on_video(video_file, output_file, all_features,
                           frame_count=None):
    """
    Draw pre-computed features on video. Allows you to process filters in
    between acquisition and plotting.

    Parameters
    ----------
    video_file: str
    output_file: str
    all_features:
    frame_count: (optional) int


    """
    video_capture = cv2.VideoCapture(video_file)
    dimensions = (int(video_capture.get(3)), int(video_capture.get(4)))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('temp.avi', fourcc, fps, dimensions)
    iframe = 0
    for face_landmarks_list in tqdm(all_features):
        iframe += 1
        if not (frame_count and (iframe > frame_count)):
            success, frame = video_capture.read()
            frame2 = draw_features(frame, face_landmarks_list)
            out.write(np.array(frame2))
    out.release()
    system(
        'ffmpeg -i temp.avi -i {} -vcodec h264 -map 0:0 -map 1:1 -shortest {} -y'
        .format(video_file, output_file))
    system('rm temp.avi')


def combine_feature_data(data):
    data2 = {key: [] for key in data[0][0]}
    for t in data:
        if t:
            f = t[0]
            for key in f:
                data2[key].append(f[key])
        else:
            for key in f:
                data2[key].append([(np.nan, np.nan)] * len(data2[key][-1]))

    data2 = {key: np.array(data2[key]) for key in data2}

    return data2


def filter_features(data2, median_win=5, gaussian_std=1):
    for key in data2:
        data = medfilt(data2[key], [median_win, 1, 1])
        data2[key] = gaussian_filter1d(data, gaussian_std, axis=0,
                                       mode='nearest')

    return data2


def uncombine_feature_data(data2):
    data3 = []
    for i in range(len(data2['bottom_lip'])):
        data3.append([{key: [tuple(x) for x in data2[key][i]]
                       for key in data2}])
    return data3


def split_video_n(input_fname, n=2, output_fname=None):
    duration = get_video_duration(input_fname)
    chunk_duration = duration / n
    return split_video_time(input_fname, chunk_duration, output_fname=output_fname)


def split_video_time(input_fname, chunk_duration=600, output_fname=None):
    """
    split video into time segments. Default = 10 minute chunks
    """
    if output_fname is None:
        output_fname = input_fname

    dest_dir = os.path.join(os.path.split(input_fname)[0], 'temp')
    os.makedirs(dest_dir, exist_ok=True)

    duration = get_video_duration(input_fname)
    output_paths = []
    for i, start in enumerate(np.arange(0, duration, chunk_duration)):
        if start + chunk_duration <= duration:
            this_chunk_duration = chunk_duration
            t_arg = '-t'
        else:
            this_chunk_duration = ''
            t_arg = ''
        this_output_fname =  '{}_p{}{}'.format(output_fname[:-4], i,
        output_fname[-4:])
        this_output_path = os.path.join(dest_dir, this_output_fname)
        output_paths.append(this_output_path)
        os.system('ffmpeg -i "{}" -ss {} {} {} -c copy "{}"'
        .format(input_fname, start, t_arg, this_chunk_duration, this_output_path))

    return output_paths


def get_video_duration(input_fname):
    """
    return duration of video in seconds
    """
    result = subprocess.check_output('ffprobe -v error -show_entries format=duration \
-of default=noprint_wrappers=1:nokey=1 "{}"'.format(input_fname), shell=True)
    return float(result.strip())


def convert_vcodec(input_fname, output_filename, vcodec='mpeg4'):
    if check_vcodec(input_fname).decode() == vcodec:
        return input_fname
    print('converting to mpeg4')
    os.system('ffmpeg -i "{}" -vcodec {} "{}"'.format(input_fname, vcodec,
              output_filename))
    print('conversion complete.')
    return output_filename


def check_vcodec(input_fname):
    result = subprocess.check_output('ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
                                     -of default=noprint_wrappers=1:nokey=1 "{}"'.format(input_fname), shell=True)
    return result.strip()
