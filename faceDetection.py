import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.compat.v1.disable_eager_execution()

# Cargar el modelo
model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))
graph = tf.compat.v1.get_default_graph()

# Inicializar el detector de caras (MTCNN)
face_detector = MTCNN()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar cámara
    ret, frame = cap.read()

    # Detección de caras con MTCNN
    faces = face_detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        
        # Dibujar un rectángulo verde alrededor de la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_array = np.expand_dims(face_img, axis=0)
        face_img_array = preprocess_input(face_img_array)

        with graph.as_default():
            predictions = model.predict(face_img_array)

        # Obtener las etiquetas de ImageNet
        labels = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

        label = labels[0][0][1]

        cv2.putText(frame, f"Clase: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
