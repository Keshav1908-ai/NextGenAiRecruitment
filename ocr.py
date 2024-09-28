import cv2
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load the digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def find_digits(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def recognise(contours, original_image):
    recognized_digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit = original_image[y:y + h, x:x + w]
        digit = cv2.resize(digit, (8, 8))  
        digit = digit.flatten() / 16.0  
        digit = np.array([digit])  
        
        prediction = knn.predict(digit)
        recognized_digits.append((prediction[0], (x, y, w, h)))  
    
    return recognized_digits

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return
    
    print("Image loaded successfully.")
    processed_image = process(image)
    
    contours = find_digits(processed_image)

    recognized_digits = recognise(contours, processed_image)
    
    for digit, (x, y, w, h) in recognized_digits:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Recognized Digits', image)
    print("OCR Reading Successful.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main(r'D:\image1.jpg')