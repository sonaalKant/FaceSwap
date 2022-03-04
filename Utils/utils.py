import cv2
import dlib
from imutils import face_utils
import numpy as np
import os

p = "/Users/sonaal/Downloads/FaceSwap/Utils/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def visualize_pts(img, pts, color=(0,255,0)):
    imgv = img.copy()
    for p in pts:
        cv2.circle(imgv, (int(p[0]), int(p[1])), 2, color, -1)
    return imgv

def visualize_triangles(img, triangles, color=(255,255,255)):
    imgv = img.copy()
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        cv2.line(imgv, pt1, pt2, color, 1)
        cv2.line(imgv, pt2, pt3, color, 1)
        cv2.line(imgv, pt3, pt1, color, 1)
    return imgv

def visualize_bbox(img, bbox, color=(0,0,255)):
    imgv = img.copy()
    x1,y1,x2,y2 = bbox
    cv2.rectangle(imgv, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return imgv

def read_video(video_path):
    path = "/Users/sonaal/Downloads/FaceSwap/viz/temp"
    os.system(f"ffmpeg -i {video_path} -s 480x640 {path}/%04d.jpg")
    

def save_video(frames_dir):
    pass

def get_bbox(pts):
    x1,y1 = pts.min(0)
    x2,y2 = pts.max(0)
    return [x1,y1,x2,y2]

def get_pts(gray, num_faces):
    rects = detector(gray, 0)
    shape = [predictor(gray, bbox) for bbox in rects]
    pts = np.array([face_utils.shape_to_np(s) for s in shape])
    # bbox = np.array([face_utils.rect_to_bb(rect) for rect in rects])
    # bbox = [ (box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in bbox]
    bbox = np.array([get_bbox(p) for p in pts])
    return pts[:num_faces] , bbox[:num_faces]

def get_triangles(pts, bbox):
    subdiv  = cv2.Subdiv2D(tuple(bbox))
    for p in pts :
        subdiv.insert(tuple(p))
    triangleList = subdiv.getTriangleList()
    return triangleList

def filter_triangles(triangles, shape):
    H,W,_ = shape
    TL = triangles.reshape(triangles.shape[0],3,2)
    c1 = set(np.where(TL[:,:,0] > W)[0])
    c1 = c1.union(set(np.where(TL[:,:,0] < 0)[0]))
    c1 = c1.union(set(np.where(TL[:,:,1] < 0)[0]))
    c1 = c1.union(set(np.where(TL[:,:,1] > H)[0]))
    
    TL = TL[~np.isin(np.arange(len(TL)), list(c1))]
    TL = TL.reshape(len(TL), -1)
    return TL

def get_correspondence(tl1, tl2, pts1, pts2):
    tl1_ = tl1.reshape(len(tl1)*3,2).astype(np.int32)
    tl2_ = tl2.reshape(len(tl2)*3,2).astype(np.int32)

    m1 = {tuple(p) : i for i,p in enumerate(pts1)}
    m2 = {tuple(p) : i for i,p in enumerate(pts2)}

    assert len(m1) == len(m2), "something is wrong"

    inv_m1 = {v : k for k,v in m1.items()}
    inv_m2 = {v : k for k,v in m2.items()}

    tl1_m = np.array([m1[tuple(t)] for t in tl1_]).reshape(len(tl1),3)
    tl2_m = np.array([m2[tuple(t)] for t in tl2_]).reshape(len(tl2),3)
    
    final_t = list()

    for t in tl1_m:
        if t in tl2_m:
            final_t.append(t)
    

    new_tl1 = np.array([ [inv_m1[t] for t in tt] for tt in final_t]).reshape(len(final_t), 6)
    new_tl2 = np.array([ [inv_m2[t] for t in tt] for tt in final_t]).reshape(len(final_t), 6)

    return new_tl1, new_tl2

def warp_image(tl1, tl2, im1, im2):
    
    im1_coords = get_grid(im1.shape)
    im2_coords = get_grid(im2.shape)

    for t1, t2 in zip(tl1, tl2):
        mat_1 = get_matrix(t1)
        mat_2 = get_matrix(t2)

        bary_coords_1 = np.linalg.inv(mat_1) @ im1_coords

        mapped_coords = mat_2 @ bary_coords_1



if __name__ == '__main__':
    img = cv2.imread("/Users/sonaal/Downloads/FaceSwap/Data/noah-centineo-1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts, bbox = get_pts(gray, 1)
    T = get_triangles(pts[0], bbox[0])
    
    imgv = visualize_bbox(img, bbox[0])
    imgv = visualize_pts(imgv, pts[0])
    imgv = visualize_triangles(imgv, T)

    cv2.imwrite("check.jpg", imgv)




