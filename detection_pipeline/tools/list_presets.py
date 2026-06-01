import keras_cv
ks = list(keras_cv.models.YOLOV8Detector.presets.keys())
with open(r"C:\Users\v-khmakhija\AppData\Local\Temp\presets.txt", "w") as f:
    f.write("\n".join(ks))
print("WROTE", len(ks))
