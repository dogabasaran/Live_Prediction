import time
import cv2
import netmodel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
TODO: Print out plot of confidence on the frame as a plot
"""


def prediction(model):
    # Load pretrained weights onto model
    ckpt_path = "training_1/cp.ckpt"
    model.load_weights(ckpt_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, pt1=(170,90),pt2=(470,390),
                color=(0,255,0), thickness=1)
        pred_img = gray[90:390, 170:470]
        ####
        ####
        pred_img = cv2.resize(pred_img, (30,30),
                interpolation=cv2.INTER_AREA)
        pred_img = pred_img / 255.0
        # Convert input image into tensor
        X = tf.convert_to_tensor([np.expand_dims(pred_img,-1)])
        # Predict input image
        prediction_array = model.predict(X)

        # Argmax prediction to get most likely label
        argmaxprediction = np.argmax(prediction_array)
        # Put prediction text on live feed display
        cv2.putText(frame, "Prediction: " + str(argmaxprediction), (200,85), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(frame, "prediction_array: {}".format(np.squeeze(prediction_array)), (180,380),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, color=(255,255,255), thickness=1)
        
        cv2.imshow("Feed", frame)
        time.sleep(.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    model = netmodel.buildModel()
    prediction(model)
