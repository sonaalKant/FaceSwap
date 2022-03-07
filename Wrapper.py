from Utils.utils import *

import numpy as np
import cv2
import argparse
import os
import glob
import tqdm

def swap_with_image(frames_dir, image_path):
	
	save_frames_dir = "viz/temp_warped/"
	if not os.path.isdir(save_frames_dir):
		os.makedirs(save_frames_dir)

	frames = glob.glob(frames_dir + "/*.jpg")
	img = cv2.imread(image_path)
	img = cv2.resize(img, (480,640))
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	pts1, bbox1 = get_pts(img_gray, 1)
	triangleList1 = get_triangles(pts1[0], bbox1[0])

	triangleList1 = filter_triangles(triangleList1, img.shape)

	for frame_path in tqdm.tqdm(frames):
		frame_name = frame_path.split("/")[-1]
		frame = cv2.imread(frame_path)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		pts2, bbox2 = get_pts(frame_gray, 1)
		triangleList2 = get_triangles(pts2[0], bbox2[0])

		triangleList2 = filter_triangles(triangleList2, frame.shape)

		tl1, tl2 = get_correspondence(triangleList1, triangleList2, pts1[0], pts2[0])

		c1 = visualize_triangles(img, tl1)
		c2 = visualize_triangles(frame, tl2)

		cv2.imwrite("check.jpg", c1)
		cv2.imwrite("check1.jpg", c2)

		warped_image = warp_image(tl1, tl2, img, frame)

		cv2.imwrite(f"{save_frames_dir}/{frame_name}", warped_image)

	import pdb;pdb.set_trace()
	

def swap(frames_dir):
	pass

def main():
	"""
	Inputs: 
	None
	Outputs:
	Swap Face
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video_path', default='/Users/sonaal/Downloads/FaceSwap/Data/data1.mp4', help='Base path of video')
	Parser.add_argument('--image_path', default='/Users/sonaal/Downloads/FaceSwap/Data/noah-centineo-1.jpg', help='Base path celeb image')
	Parser.add_argument('--save_name', default="check.mp4", help="name of saved video")


	Args = Parser.parse_args()
	video_path = Args.video_path
	image_path = Args.image_path

	frames_dir = "/Users/sonaal/Downloads/FaceSwap/viz/temp"

	# read_video(video_path)

	if image_path is not None:
		new_frames_dir = swap_with_image(frames_dir, image_path)
	else:
		new_frames_dir = swap(frames_dir)
	
	save_video(new_frames_dir)

	
if __name__ == '__main__':
	main()