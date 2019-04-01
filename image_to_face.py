import dlib
import matplotlib.pyplot as plt
import os
import glob
from imutils import face_utils
from imutils.face_utils import FaceAligner
import cv2

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200) 

# Load image from directory and write the face to the same director
img_no = 1
base_dir = "/home/mukul/Desktop/machine_learning/deepLearning/celebrity_face_recognition/face_images"
for root, dir, files in os.walk(base_dir):
    #os.path.splitext(os.path.basename(root))[-1]
    if 'cropped' not in str(os.path.splitext(os.path.basename(root))[0]):
        if '_' not in str(os.path.splitext(os.path.basename(root))[0]):
            print('entered')
            FACE_DIR = base_dir + "/" + "cropped" + "/" + str(os.path.splitext(os.path.basename(root))[0])
            create_folder(FACE_DIR)
            for file in files:
                image = root + "/" + file
                img = cv2.imread(image)
                try:
                    img.shape
                    print("checked for shape".format(img.shape))
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector(img_gray)
                    if len(faces) == 1:
                        face = faces[0]
                        (x, y, w, h) = face_utils.rect_to_bb(face)
                        face_img = img_gray[y-50:y + h+100, x-50:x + w+100]
                        face_aligned = face_aligner.align(img, img_gray, face)

                        face_img = face_aligned
                        img_path = FACE_DIR +"/"+ str(img_no) + ".jpg"
                        cv2.imwrite(img_path, face_img)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
                        #cv2.imshow("aligned", face_img)
                        img_no += 1
                except AttributeError:
                    print("shape not found")        

