import cv2
import os
import time
import shutil

'''
collected images will be 300x300
'''


def main(MAKE_NEW_DATASET, IMAGES_PER_LABEL, LABELS):
    if MAKE_NEW_DATASET:
        shutil.rmtree('data')
        os.makedirs('data')
        for label in LABELS:
            try:
                os.makedirs(os.path.join('data', str(label)))
            except FileExistsError:
                print("Directory {} already exists".format(os.path.join('data', str(label))))
            cap = cv2.VideoCapture(0)
            count = 0
            save_count = 0
            # countdown(down_from=3)
            countdown = True
            displaycount = int(time.time()) + 5

            while True:
                # Capture frame by frame
                ret, frame = cap.read()
                # Convert to gray
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Draw rectangle to capture data in
                cv2.rectangle(gray, pt1=(170, 90), pt2=(470, 390),
                                color=(0, 255, 0), thickness=1)
                # Don't show display count if countdown has reached zero
                if displaycount - int(time.time()) <= 0:
                    countdown=False
                # Display countdown on frame if countdown is active
                if countdown:
                    cv2.putText(gray, str(displaycount - int(time.time())), (320,240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                # Display image
                cv2.imshow('Capturing data for image: {}'.format(label), gray)
                # Save image every 4 frames up if we still need data, and countdown is finished
                if count % 4 == 0 and save_count < IMAGES_PER_LABEL and not countdown:
                    cv2.imwrite(os.path.join('data', str(label), str(save_count) + '.jpg'),
                                            gray[90:390,170:470])
                    print("Captured image data for: {} as {}.jpg".format(label, save_count))
                    save_count += 1
                elif save_count >= IMAGES_PER_LABEL:
                    print("Taken enough data, now will take in next label's data")
                    break
                count += 1
                # Quit if "q" is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Make True if you want to make a new dataset
    MAKE_NEW_DATASET = True
    # Identify the labels
    labels = [0, 1]
    # Number of images to be taken for each label. Allows for balanced dataset
    IMAGES_PER_LABEL = 20
    main(MAKE_NEW_DATASET, IMAGES_PER_LABEL, labels)
