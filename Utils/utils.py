import cv2
import dlib
from imutils import face_utils

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def read_video(video_path):
    pass

def save_video(frames_dir):
    pass

def get_face(gray, num_faces):
    rects = detector(gray, 1)
    import pdb;pdb.set_trace()
    return rects[:num_faces]

def get_pts(gray, bbox):
    shape = predictor(gray, bbox)
    import pdb;pdb.set_trace()
    shape = face_utils.shape_to_np(shape)
    return shape

def get_triangles(bbox, pts):
    pass


if __name__ == '__main__':
    img = cv2.imread("/vulcanscratch/sonaalk/FaceSwap/Data/noah-centineo-1.jpg", 0)
    rect = get_face(img, 2)
    get_pts(img, rect)


