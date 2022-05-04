from torchreid.utils import FeatureExtractor
import cv2


cap = cv2.VideoCapture('videos/init/Double1.mp4')
while True:
	success, img = cap.read()
	if success != True:
		cap.release()
		print('THE END.....')
		break
	features = extractor(image_list)
	print(features.shape)