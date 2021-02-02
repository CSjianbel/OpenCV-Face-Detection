import os
import sys
import argparse

try:
    import cv2

except ModuleNotFoundError:
    print("Required Modules Not Installed...")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

FACE_CASCADE_PATH = os.path.join("data", "haarcascade_frontalface_default.xml")

def main():

    # Parse Command Line Arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Set mode to Detect face in [image, video, webcam] : DEFAULT: webcam", type=str, default="webcam")
    parser.add_argument("-p", "--path", help="Path to image or video file requires --mode to be set to image or video", type=str)
    args = parser.parse_args()

    # mode : function
    MODES = {
        "webcam": Webcam,
        "image": Image,
        "video": Video
    } 

    if not args.mode in MODES.keys():
        err_exit(f"Invalid Mode! : Options: {list(MODES.keys())}", 2)

    elif args.mode == "webcam":
        MODES[args.mode]()

    else:
        # Verify Path
        if not args.path:
            err_exit("Please provide a path for image or video when mode is set to [image, video]", 3)
        
        if not verifyPath(args.path):
            err_exit(f"Invalid path! : {path}", 4)

        MODES[args.mode](args.path)


def Webcam():
    """
    Detects a face in a webcam
    Params: None
    Return: None
    """
    print("in webcam func")


def Image(path: str):
    """
    Detects a face in a given Image
    Params: str
    Return: None
    """

    # Reads the image as a numpy array
    img = cv2.imread(os.path.join(".", path), 1)
    img_filename = path.split('/')[-1].split('.')[0]

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Creates a window that shows the image
    cv2.imshow(img_filename, img)

    # Wait for keypress
    cv2.waitKey(0)

    # Destroys all windows open
    cv2.destroyAllWindows()


def Video(path: str):
    """
    Detects a face in a given Video
    Params: str
    Return: None
    """
    print("IN VIDEO FUNC")


def verifyPath(path: str):
    """
    Verifies if path to image or Video exists
    Params: str
    Return: bool
    """
    try:
        with open(path) as f:
            return True
    except FileNotFoundError:
        return False


def err_exit(msg: str, code: int):
    """
    Prints out an error message and exits the program with code
    Params: str, int
    Return: None
    """
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()