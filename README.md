# face extraction


## Installation
I recommend using a conda envoronment.
`ffmpeg` and `dlib` are required. If on mac, install `ffmpeg` by running `brew install ffmpeg`. 
It also uses `opencv3`, which can be installed using `conda install opencv3`, 
however I was only able to get this to work in python 3.5, so I used conda 
virtual environment. Putting it all together (assuming anaconda is properly
installed):
```bash
brew install ffmpeg
conda create -n face_extraction_env python=35 opencv3 pillow numpy
source activate face_extraction_env
pip install tqdm
pip install face_recognition
```
## Usage
```python
from face_extraction import detect_faces_video
detect_faces_video('video_in.mp4', 'video_out.avi')
```

OpenCV is pretty picky about the form of the input video, so you may 
have to use `ffmpeg` to change the video codec.

## Ideas
* use mixture of median filter and gaussian filter on points to improve tracking
* Sometimes a second face appears in subject's clothing. Figure out how to remove this.
* Implement in C++
