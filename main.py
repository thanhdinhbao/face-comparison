import cv2
import dlib
import numpy as np

#@ThanhDB (31/03/2023)

# Load file ảnh
img1 = cv2.imread("face3.jpg")
img2 = cv2.imread("face4.jpg")

# Khởi tạo nhận diện khuân mặt của thư viện dlib
detector = dlib.get_frontal_face_detector()

# Nhận diện khuân mặt trong ảnh
faces1 = detector(img1, 1)
faces2 = detector(img2, 1)

# Khởi tạo model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Tính toán khuân mặt
face1_encoding = None
face2_encoding = None

if len(faces1) > 0:
    face1_landmarks = predictor(img1, faces1[0])
    face1_encoding = facerec.compute_face_descriptor(img1, face1_landmarks)

if len(faces2) > 0:
    face2_landmarks = predictor(img2, faces2[0])
    face2_encoding = facerec.compute_face_descriptor(img2, face2_landmarks)

# So sánh chi tiết khuân mặt 
if face1_encoding is not None and face2_encoding is not None:
    distance = np.linalg.norm(np.array(face1_encoding) - np.array(face2_encoding))
    print("Độ khớp giữa hai khuân mặt là:", distance*100)
