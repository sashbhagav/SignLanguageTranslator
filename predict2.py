import sys
from google.cloud import vision
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
import cv2
import numpy as np
import tensorflow as tf
import copy
import os
import matplotlib.pyplot as plt

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned


if __name__ == '__main__':
	project_id = "signlanguage-5" #constants
	model_id = "handsigns_v20190126223903"

	cap = cv2.VideoCapture(0)
	while True :
		ret, frame = cap.read()
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
    	
    	go = str(frame.tostring())
    	#sys.stdout.write(str(frame.tostring()))
    	
    	#file_path = sys.argv[1]
    	#project_id = sys.argv[2]
    	#model_id = sys.argv[3]

		with open(go, 'rb') as ff:
		    content = ff.read()
			print get_prediction(content, project_id,  model_id)

cap.release()                                                                  
cv2.destroyAllWindows()
