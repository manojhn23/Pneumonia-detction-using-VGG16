import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model

def check(res):
    p2=["normal","pneumonia"]
    path=p2
    model = load_model('chest_xray.h5',compile=False)
    pred=model.predict(res)
    res=np.argmax(pred)
    res=path[res]
    print(res)

def convert_img_to_tensor2(fpath):
    img = cv2.imread(fpath)
    img = cv2.resize(img,(224,224))
    res = img_to_array(img)
    res = np.array(res, dtype=np.float16)/ 255.0
    res = res.reshape(-1,224,224,3)
    res = res.reshape(1,224,224,3)
    return res

t2="C:\\Users\\manoj\\PycharmProjects\\python1\\Project\\static\\img.jpg"

res=convert_img_to_tensor2(t2)

check(res)
