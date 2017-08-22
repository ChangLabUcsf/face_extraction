from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
from os import system
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
import pickle


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


def detect_faces_video(video_file, output_video_file=None, fps=None,
                       dimensions=None, frame_count=None,
                       output_data_file=None):
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
    frame_count: int or None
        Only take the first `frame_count` frames. If None, use all frames.

    Returns
    -------
    list(
        dict(
            feature:np.array()
            ),...
        )

    """
    video_capture = cv2.VideoCapture(video_file)
    if frame_count is None:
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if dimensions is None:
        dimensions = (int(video_capture.get(3)), int(video_capture.get(4)))

    if output_video_file:
        if fps is None:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('temp.avi', fourcc, fps, dimensions)

    all_features = []
    for _ in tqdm(range(frame_count)):
        success, frame = video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        all_features.append(face_landmarks_list)

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
