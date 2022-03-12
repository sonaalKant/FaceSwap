import cv2
import dlib
from imutils import face_utils
import numpy as np
import os
from scipy.spatial import distance

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

    idxs = np.array([m1[tuple(t)] for t in tl1_]).reshape(len(tl1),3)

    new_tl1 = np.array([ [pts1[t] for t in tt] for tt in idxs]).reshape(len(idxs), 6)
    new_tl2 = np.array([ [pts2[t] for t in tt] for tt in idxs]).reshape(len(idxs), 6)

    return new_tl1, new_tl2


def get_grid(t1):
    xmin, ymin = t1.min(0)
    xmax, ymax = t1.max(0)
    xv, yv = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    coords = np.concatenate([xv.reshape(-1,1), yv.reshape(-1,1)], axis=1)
    coords = np.concatenate([coords, np.ones((len(coords), 1))], axis=-1)
    return coords

def remove_invalid(coords):
    c1 = set(np.where(coords[:,0] < 0)[0])
    c1 = c1.union(set(np.where(coords[:,1] < 0)[0]))
    c1 = c1.union(set(np.where(coords[:,2] < 0)[0]))
    c1 = c1.union(set(np.where(coords[:,0] > 1)[0]))
    c1 = c1.union(set(np.where(coords[:,1] > 1)[0]))
    c1 = c1.union(set(np.where(coords[:,2] > 1)[0]))
    
    coords = coords[~np.isin(np.arange(len(coords)), list(c1))]
    if len(coords) == 0:
        return coords, False, list(c1)

    coords = coords.reshape(len(coords), -1)
    return coords, True, list(c1)

def bilinear_interpolate(img, coords):
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def warp_image(tl1, tl2, im1, im2):

    result = im2.copy()
    for t1, t2 in zip(tl1, tl2):
        # print(t1,t2)
        t1 = t1.reshape(-1, 2)
        t2 = t2.reshape(-1, 2)

        coords1 = get_grid(t2)

        mat_1 = np.concatenate([t1, np.ones((3,1))], axis=-1).T
        mat_2 = np.concatenate([t2, np.ones((3,1))], axis=-1).T

        try:
            bary_coords_1 = (np.linalg.inv(mat_2) @ coords1.T).T
        except:
            continue

        bary_coords_1, valid, ignore_idxs = remove_invalid(bary_coords_1)

        if valid:
            coords1 = coords1[~np.isin(np.arange(len(coords1)), list(ignore_idxs))]
            mapped_coords = (mat_1 @ bary_coords_1.T)
            x = coords1[:,0].astype(np.int32)
            y = coords1[:,1].astype(np.int32)
            result[y,x] = bilinear_interpolate(im1, mapped_coords[:2,:])
        # else:
        #     print("Invalid")
    
    return result

'''
Reference :  https://khanhha.github.io/posts/Thin-Plate-Splines-Warping/
'''
def warp_image_TSP(pts1, pts2, im1, im2):
    H,W = im2.shape[:2]
    result = im2.copy()
    
    # import pdb;pdb.set_trace()
    # K1 = distance.cdist(pts2, pts2, lambda u,v : np.sum((u-v)**2) * np.log(np.sum((u-v)**2) + 1e-8) )
    K = distance.cdist(pts2, pts2, 'sqeuclidean')
    K = K*np.log(K + 1e-8)
    P = np.concatenate([np.ones((len(pts2), 1)), pts2], axis=1)
    zeros = np.zeros((3, 3))

    top = np.concatenate([K, P], axis=1)
    bot = np.concatenate([P.T, zeros], axis=1)
    lamb = 1e-3

    mat = np.concatenate([top, bot], axis=0) + lamb * np.eye(len(pts2)+3, len(pts2)+3)
    mat_inv = np.linalg.inv(mat)

    # x axis
    v = np.concatenate([pts1[:,:1], np.zeros((3,1))], axis=0)
    x_params = mat_inv @ v

    #y axis
    v = np.concatenate([pts1[:,1:2], np.zeros((3,1))], axis=0)
    y_params = mat_inv @ v

    box = get_bbox(pts2)
    coords = get_grid(np.array(box).reshape(2,2))
    x = coords[:,0].astype(np.int32)
    y = coords[:,1].astype(np.int32)
    coords = np.roll(coords, 1)

    #K = distance.cdist(coords[:,1:], pts2, lambda u,v : np.sum((u-v)**2) * np.log(np.sum((u-v)**2) + 1e-8) )
    K = distance.cdist(coords[:,1:], pts2, 'sqeuclidean')
    K = K*np.log(K + 1e-8)
    M = np.concatenate([K,coords], axis=1)

    xs = M @ x_params
    ys = M @ y_params

    mapped_coords = np.concatenate([xs, ys], axis=1)
    
    result[y,x] = bilinear_interpolate(im1, mapped_coords.T)
    
    return result


if __name__ == '__main__':
    img = cv2.imread("/Users/sonaal/Downloads/FaceSwap/Data/noah-centineo-1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts, bbox = get_pts(gray, 1)
    T = get_triangles(pts[0], bbox[0])
    
    imgv = visualize_bbox(img, bbox[0])
    imgv = visualize_pts(imgv, pts[0])
    imgv = visualize_triangles(imgv, T)

    cv2.imwrite("check.jpg", imgv)




