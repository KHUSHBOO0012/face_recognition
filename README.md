# face recognition for Indian Celebrity

## Models Used:

   - pretrained [`inception_v2`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/inception_blocks_v2.py) as [model](https://github.com/KHUSHBOO0012/face_recognition/tree/master/weights) which takes (3,96,96) dimension input images and gives 128 dimension output embeddings
   - pretrained [`inception_resnet_v1`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/inception_resnet_v1.py) as [model1](https://github.com/KHUSHBOO0012/face_recognition/blob/master/facenet_keras_weight.h5) which takes (160,160,3) dimensional input images and gives 128 dimensional output vectors.
   
## Files Description

- [`face_images`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/face_images): Contains cropped and uncropped images of celebrity arranged according to name of the celebrity.

- [`images`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/images): Contains celebrity images arranged in number 
in order to compare the test images from database.

- [`images1`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/images1): Contains 6 images to fintune the threshhold and debug code fastly.

- [`test_images`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/test_images): Contain just one image to test the output of the model

- [`weights`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/weights): contains the weight of pretrained [`inception_v2`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/inception_blocks_v2.py) in csv files

- [celebrity_face_recognition.ipynb](https://github.com/KHUSHBOO0012/face_recognition/blob/master/celebrity_face_recognition.ipynb): notebook containing the codes with slight explanation.

- [`celebrity_face_recognition.py`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/celebrity_face_recognition.py) python file for the ipynb file provided above.

- [`facenet_keras_weight.h5`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/facenet_keras_weight.h5): pretrained weight of [`inception_resnet_v1`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/inception_resnet_v1.py).

- [`image_to_face.py`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/image_to_face.py) : Converts uncropped imges to the cropped image.

- [`video_to_face.py`](https://github.com/KHUSHBOO0012/face_recognition/blob/master/video_to_face.py): Convert the uncropped images from webcam to cropped and same them to folder [`images`](https://github.com/KHUSHBOO0012/face_recognition/tree/master/images). 
