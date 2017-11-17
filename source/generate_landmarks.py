import sys
import os
import dlib
import glob
from skimage import io

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "../data/pics"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])


my_glob = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
my_glob.sort(key=sortKeyFunc)
print(my_glob)
for f in my_glob:
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        file_write = ""
        for i in range(0, 68):
            file_write += str(shape.part(i).x) + ", " + str(shape.part(i).y) + ", "
        with open("../data/landmarks.txt", "a") as f:
            f.write(file_write)
            f.write("\n")
