import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
from google.protobuf.json_format import MessageToJson

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
#check, frame = cam.read()
img_counter = 0

#method to get prediction statistics
def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned



#lines 47-70 are for searching through folder to find .png
path = '/Users/sashreek/Documents'
folder = os.fsencode(path)
filenames = []
THRESHOLD_VALUE = 200

def loadImages(path):
    image_files = sorted(
                         [os.path.join(path, 'SignLanguage', file) for file in os.listdir(path + "/SignLanguage") if file.endswith('.png')]
                         )
        
                         
                         count = 0;
                         #going through for loop to do color degrading and convert to jpg
                         for img in image_files:
                             bw_image = Image.open(img)
                             bw_image = bw_image.convert('L')
                             bw_image.save("result{}.jpg".format(count))
                             
                             imgData = np.asarray(bw_image)
                             thresholdedData = (imgData > THRESHOLD_VALUE) * 1.0
                             
                                 count = count + 1
                             # print(image_files)

print(count)
return image_files

if __name__ == '__main__':
    #cv2 camera, this allows for snapshots
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

cam.release()
loadImages(path)

img_count = 0
    word = ""
    #iterating through result images to get predictions stats from them
    for file in os.listdir(path + "/SignLanguage"):
        if file.endswith('.jpg'):
            file_path = file
            project_id = 'signlanguage-5'
            model_id = 'ICN3732868437682118934'
            
            with open(file_path, 'rb') as ff:
                content = ff.read()
            
            #result = MessageToDict(str(get_prediction(content, project_id,  model_id))
            result = get_prediction(content, project_id,  model_id)
            
            print('FOR ' + file + '\n')
            jsonStr = MessageToJson(result)
            jsonObj = json.loads(jsonStr)
            print('Classification:')
            if('payload' in jsonObj.keys()):
                letter = jsonObj['payload'][0]['displayName']
                print('letter predicted ----- ', letter)
                # word = word + letter
                # print('the word is ', word)
                #cv2.putText(frame, word, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                print('------------------')
            else:
                print('letter not recognized')




#print(type(result.payload))
#print(word)
#cv2.putText(frame, word)
#for response in result.payload:
#print('Predicted {}'.format(response.display_name))
cv2.destroyAllWindows()
