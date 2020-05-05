import cv2

def loadImage(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return img

def showImage(img):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resizeImage(img, scale):
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
