from Utils.utils import *

import numpy as np
import cv2
import argparse
import os
import glob
import tqdm

def swap_with_image(frames_dir, image_path, method, interp):
	
	save_frames_dir = f"viz/temp_warped/{method}/"
	
	os.system(f"rm -rf {save_frames_dir}")

	if not os.path.isdir(save_frames_dir):
		os.makedirs(save_frames_dir)

	frames = glob.glob(frames_dir + "/*.jpg")
	img = cv2.imread(image_path)
	# img = cv2.resize(img, (480,640))
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	pts1, bbox1 = get_pts(img_gray, 1)
	# import pdb;pdb.set_trace()
	triangleList1 = get_triangles(pts1[0], bbox1[0])
	triangleList1 = filter_triangles(triangleList1, img.shape)

	if interp:
		print("Interpolating ...")
		all_pts, all_boxes = get_complete_video_pts(frames, 1)
	else:
		all_pts = np.zeros((len(frames)))
		all_boxes = np.zeros((len(frames)))

	for i, (frame_path, pts2, bbox2) in tqdm.tqdm( enumerate(zip(sorted(frames), all_pts, all_boxes))):
		frame_name = frame_path.split("/")[-1]
		frame = cv2.imread(frame_path)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if not interp:
			pts2, bbox2 = get_pts(frame_gray, 1)

		if len(pts2) < 1:
			cv2.imwrite(f"{save_frames_dir}/{frame_name}", frame)
			continue

		triangleList2 = get_triangles(pts2[0], bbox2[0])
		triangleList2 = filter_triangles(triangleList2, frame.shape)
		tl1, tl2 = get_correspondence(triangleList1, triangleList2, pts1[0], pts2[0])

		# c1 = visualize_triangles(img, tl1[:5])
		# c2 = visualize_triangles(frame, tl2[:5])

		# cv2.imwrite("check.jpg", c1)
		# cv2.imwrite("check1.jpg", c2)

		if method == "TRI":
			warped_image, mask = warp_image(tl1, tl2, img, frame)
		elif method == "TSP":
			warped_image, mask = warp_image_TSP(pts1[0], pts2[0], img, frame, tl1, tl2)

		# Blending
		# kernel = np.ones((10, 10), np.uint8)
		# mask = cv2.erode(mask, kernel, iterations=1)
		##Poisson Blending
		r = cv2.boundingRect(mask[:,:,0])

		center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
		output = cv2.seamlessClone(warped_image, frame, mask, center, cv2.NORMAL_CLONE)

		cv2.imwrite(f"{save_frames_dir}/{frame_name}", output)

	return save_frames_dir
	

def swap(frames_dir, method, interp):
	
	save_frames_dir = f"viz/temp_warped_2/{method}"
	
	os.system(f"rm -rf {save_frames_dir}")

	if not os.path.isdir(save_frames_dir):
		os.makedirs(save_frames_dir)
	
	frames = glob.glob(frames_dir + "/*.jpg")

	if interp:
		print("Interpolating ...")
		all_pts, all_boxes = get_complete_video_pts(frames, 2)
	else:
		all_pts = np.zeros((len(frames)))
		all_boxes = np.zeros((len(frames)))

	for i, (frame_path, pts2, bbox2) in tqdm.tqdm( enumerate(zip(sorted(frames), all_pts, all_boxes))):
		frame_name = frame_path.split("/")[-1]
		frame = cv2.imread(frame_path)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		if not interp:
			pts2, bbox2 = get_pts(frame_gray, 2)

		if len(pts2) < 2:
			cv2.imwrite(f"{save_frames_dir}/{frame_name}", frame)
			# print(frame_name)
			continue
		
		triangleList1 = get_triangles(pts2[0], bbox2[0])
		triangleList2 = get_triangles(pts2[1], bbox2[1])

		triangleList1 = filter_triangles(triangleList1, frame.shape)
		triangleList2 = filter_triangles(triangleList2, frame.shape)

		tl1, tl2 = get_correspondence(triangleList1, triangleList2, pts2[0], pts2[1])

		# c1 = visualize_triangles(img, tl1[:5])
		# c2 = visualize_triangles(frame, tl2[:5])

		# cv2.imwrite("check.jpg", c1)
		# cv2.imwrite("check1.jpg", c2)
		
		if method == "TRI":
			warped_image1, mask1 = warp_image(tl1, tl2, frame, frame)
			warped_image2, mask2 = warp_image(tl2, tl1, frame, frame)
		else:
			warped_image1, mask1 = warp_image_TSP(pts2[0], pts2[1], frame, frame, tl1, tl2)
			warped_image2, mask2 = warp_image_TSP(pts2[1], pts2[0], frame, frame, tl2, tl1)
		
		warped_image = warped_image1 + warped_image2
		mask = mask1 + mask2

		# import pdb;pdb.set_trace()

		# Blending
		# kernel = np.ones((10, 10), np.uint8)
		# mask = cv2.erode(mask, kernel, iterations=1)
		##Poisson Blending
		r = cv2.boundingRect(mask[:,:,0])

		center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
		output = cv2.seamlessClone(warped_image, frame, mask, center, cv2.NORMAL_CLONE)

		cv2.imwrite(f"{save_frames_dir}/{frame_name}", output)

	return save_frames_dir

def main():
	"""
	Inputs: 
	None
	Outputs:
	Swap Face
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video_path', default='/Users/sonaal/Downloads/FaceSwap/Data/data2.mp4', help='Base path of video')
	Parser.add_argument('--image_path', default=None, help='Base path celeb image')
	Parser.add_argument('--save_name', default="Data1", help="name of saved video")
	Parser.add_argument('--interpolate', action='store_true', help="use interpolation to fill missing points in frame")


	Args = Parser.parse_args()
	video_path = Args.video_path
	image_path = Args.image_path
	interp = Args.interpolate


	frames_dir = "/Users/sonaal/Downloads/FaceSwap/viz/temp"

	os.system(f"rm -rf {frames_dir}")

	if not os.path.isdir(frames_dir):
		os.makedirs(frames_dir)

	read_video(video_path)

	if image_path is not None:
		new_frames_dir_TRI = swap_with_image(frames_dir, image_path, method="TRI", interp=interp)
		new_frames_dir_TSP = swap_with_image(frames_dir, image_path, method="TSP", interp=interp)
	else:
		new_frames_dir_TRI = swap(frames_dir, method="TRI", interp=interp)
		new_frames_dir_TSP = swap(frames_dir, method="TSP", interp=interp)
	
	save_video(new_frames_dir_TSP, f"Data/{Args.save_name}OutputTPS.mp4")
	save_video(new_frames_dir_TRI, f"Data/{Args.save_name}OutputTRI.mp4")

	
if __name__ == '__main__':
	main()