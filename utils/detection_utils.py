import cv2
import os

#performing preprocesing to make all images of same size and remove noise like shadows,rainy drops etc. to improve the accuracy of the model
def preprocessing(image):
    
    target_width = 800
    height, width = image.shape[:2] # take only height and width, ignore color channels
    
    if width > target_width:
        
       scale=target_width / width
       target_height = int(height * scale)
       image = cv2.resize(image, (target_width, target_height))

    # now removing noise from the image using fastNlMeansDenoisingColored function of opencv
    #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image


def detect_plate(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    plate_cascade=cv2.CascadeClassifier(
        'utils/haarcascade_russian_plate_number.xml'
        )

    plates=plate_cascade.detectMultiScale(
        image=gray,
        scaleFactor=1.015,
        minNeighbors=10,
        minSize=(65, 15)
    )
    cropped_plates = []
    result_image = image.copy()
    for (x,y,w,h) in plates:
        cv2.rectangle(
            result_image,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2)
        cropped_plate = image[y:y+h, x:x+w]    
        cropped_plates.append(cropped_plate)

        os.makedirs('croped_outputs', exist_ok=True)
        cv2.imwrite(f'croped_outputs/cropped_plate_{x}_{y}.jpg', cropped_plate)
        
    return result_image, cropped_plates

