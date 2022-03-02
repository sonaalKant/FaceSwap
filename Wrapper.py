from Utils.utils import *

import numpy as np
import cv2
import argparse
import os
import glob

def swap_with_image(frames_dir, image_path):
	frames = glob.glob(frames_dir + "/*.jpg")
	img = cv2.imread(image_path)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bbox1 = get_face(img_gray, 1)
	pts1 = get_pts(img_gray, bbox1)
	triangleList1 = get_triangles(bbox1, pts1)


	for frame_path in frames:
		frame = cv2.imread(frame_path)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		bbox2 = get_face(frame_gray, 1)
		pts2 = get_pts(frame_gray, bbox2)
		triangleList2 = get_triangles(bbox1, pts1)

		Tri_corr = get_correspondence(triangleList1, triangleList2, pts1, pts2)

	

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
	Parser.add_argument('--video_path', default='/vulcanscratch/sonaalk/Stitching/Phase2/Data/Train', help='Base path of video')
	Parser.add_argument('--image_path', default=None, help='Base path celeb image')
	Parser.add_argument('--save_name', default="check.mp4", help="name of saved video")


	Args = Parser.parse_args()
	video_path = Args.video_path
	image_path = Args.image_path

	frames_dir = read_video(video_path)

	if image_path is not None:
		new_frames_dir = swap_with_image(frames_dir, image_path)
	else:
		new_frames_dir = swap(frames_dir)
	
	save_video(new_frames_dir)

	
if __name__ == '__main__':
	main()