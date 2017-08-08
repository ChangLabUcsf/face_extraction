from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
from os import system


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
            d.line(face_landmarks[facial_feature], width=linewidth)

    if fname is None:
        return pil_image
    pil_image.format = "PNG"
    pil_image.save(fname)


def detect_faces_video(video_file, output_file=None, fps=None,
                       dimensions=None, frame_count=None):
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

    if output_file:
        if fps is None:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('temp.avi', fourcc, fps, dimensions)

    all_features = []
    for _ in tqdm(range(frame_count)):
        success, frame = video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        all_features.append(face_landmarks_list)

        if output_file:
            frame2 = draw_features(frame, face_landmarks_list)
            out.write(np.array(frame2))

    if output_file:
        out.release()
        """
        * encoding with h264 makes the video about 1/20th the size
        * h264 videos can be played in VLC
        * the `map` arguments take the audio from the original video file
        """
        system('ffmpeg -i temp.avi -i {} -vcodec h264 -map 0:0 -map 1:1 -shortest {}'
               .format(video_file, output_file))
        system('rm temp.avi')

    return all_features
