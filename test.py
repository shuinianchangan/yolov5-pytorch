# -*- coding:utf-8 -*-

from roboflow import Roboflow
rf = Roboflow(api_key="4pQeCbuHzSzTvTEifWJ1")
project = rf.workspace().project("bike-helmet-detection-2vdjo")
model = project.version(1).model

# infer on a local image
# print(model.predict("img/BikesHelmets44.png", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("img/2.png", confidence=40, overlap=30).save("img/prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())