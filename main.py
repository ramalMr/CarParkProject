import cv2
import pickle
import cvzone
import numpy as np

# Video faylını yükləyirik
cap = cv2.VideoCapture('carPark.mp4') 

# Park yerlərinin koordinatlarını saxladığımız faylı açırıq
with open('carmodul', 'rb') as f:
    posList = pickle.load(f)

# Park yerinin genişlik və hündürlüyünü müəyyənləşdiririk
width, height = 107, 48

# Park yerinin boş olub-olmadığını yoxlayan funksiya
def checkParkingSpace(imgPro):
    spaceCounter = 0

    # Bütün park yerlərini yoxlayırıq
    for pos in posList:
        x, y = pos

        # Park yerinin görüntüsünü kəsirik
        imgCrop = imgPro[y:y + height, x:x + width]
        # Kəsilmiş görüntüdəki qeyri-sıfır piksellərin sayını hesablayırıq
        count = cv2.countNonZero(imgCrop)

        # Əgər qeyri-sıfır piksel sayı azdırsa, park yerinin boş olduğunu farz edirik
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        # Park yerinin üzərinə dikdörtgən çəkirik
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        # Piksel sayını yazdırırıq
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Boş park yerlərinin sayını ekranda göstəririk
    cvzone.putTextRect(img, f'Park ucun Uygun Olan: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0,200,0))

# Ana döngü - videoyu frame frame oxuyaraq işləyirik
while True:

    # Video sona çatsa, başa qaytarırıq
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Video faylından yeni frame oxuyuruq
    success, img = cap.read()
    # Frame'i gri skala çeviririk
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Görüntünü yumşaltmaq üçün Gaussian Blur tətbiq edirik
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # Görüntünü ikili (binary) şəkilə çeviririk
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    # Kenarları yumşaltmaq üçün median blur tətbiq edirik
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    # Görüntünü genişləndirmək üçün dilate tətbiq edirik
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Park yerlerini yoxlayırıq
    checkParkingSpace(imgDilate)
    # Sonuçları ekranda göstəririk
    cv2.imshow("Image", img)
    # 10 milisaniyə gözləyirik
    cv2.waitKey(10)
