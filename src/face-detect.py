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
MAX_HEIGHT, MAX_WIDTH = 600, 800


def main():
    """
    Main function of face detection using OpenCV
    """
    # Parse Command Line Arguments 
    args = parseArgs()

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
            err_exit(f"Invalid path! : {args.path}", 4)

        MODES[args.mode](args.path)


def Webcam():
    """
    Detects a face in a webcam
    Params: None
    Return: None
    """
    # Print out instructions to user
    print("Press '0' to stop capturing on webcam...")
    # Capture video on webcam
    video = cv2.VideoCapture(0)

    # Read face cascade xml
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    while True:
    
        # Turn on webcam and read data
        check, frame = video.read()

        # Detect faces in image
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5)
        print(f"Found {len(faces)} faces on webcam!")

        # Draw rectangles around the detected faces
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing", frame)

        # Press 0 to end video capturing
        if cv2.waitKey(1) == ord('0'):
            print("Webcam capturing ended...")
            break
            
    # Release the VideoCapture object  
    video.release()


def Image(path: str):
    """
    Detects a face in a given Image
    Params: str
    Return: None
    """
    # Reads the image as a numpy array
    img = cv2.imread(os.path.join(".", path))
    if not img.any():
        err_exit("Invalid file! - please provde a path to an image", 404)

    img_filename = path.split('/')[-1].split('.')[0]

    # Dynamically resize image to stay within the bounds of MAX_HEIGHT and MAX_WIDTH
    while True:
        height, width = img.shape[:2]
        if height > MAX_HEIGHT:
            img = ResizeWithAspectRatio(img, height=MAX_HEIGHT)
        elif width > MAX_WIDTH:
            img = ResizeWithAspectRatio(img, width=MAX_WIDTH)
        
        height, width = img.shape[:2]
        if height <= MAX_HEIGHT and width <= MAX_WIDTH:
            break

    # Read face cascade xml
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    # Detect faces in image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

    print(f"Found {len(faces)} faces on {path}")

    # Put rectangles around detected faces in image
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Creates a window that shows the image
    cv2.imshow(img_filename, img)

    # Wait for keypress
    cv2.waitKey(0)
    print("Press any key to continue...")

    # Destroys all windows open
    cv2.destroyAllWindows()


def Video(path: str):
    """
    Detects a face in a given Video
    Params: str
    Return: None
    """
    # Print out instructions to user
    print("Press '0' key to stop playing video...")
    # To capture video from existing video.   
    video = cv2.VideoCapture(path)
    if not video:
        err_exit("Invalid file! - please provde a path to an image", 404)
  
    # Load the cascade  
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)  

    # Get filename 
    img_filename = path.split('/')[-1].split('.')[0]
    
    while True:  
        # Read the frame  
        check, frame = video.read()  

        # Dynamically resize image to stay within the bounds of MAX_HEIGHT and MAX_WIDTH
        while True:
            height, width = frame.shape[:2]
            if height > MAX_HEIGHT:
                frame = ResizeWithAspectRatio(frame, height=MAX_HEIGHT)
            elif width > MAX_WIDTH:
                frame = ResizeWithAspectRatio(frame, width=MAX_WIDTH)
            
            height, width = frame.shape[:2]
            if height <= MAX_HEIGHT and width <= MAX_WIDTH:
                break

        # Convert to grayscale  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
        # Detect the faces on a given frame  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        print(f"Found {len(faces)} faces!")
    
        # Draw the rectangle around detected faces
        for (x, y, w, h) in faces:  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
    
        # Display the video 
        cv2.imshow('Video', frame)  
    
        # Stop playing video if '0' key is pressed  
        if cv2.waitKey(1) == ord('0'):
            print("Video playing and face detection stopped...")
            break
            
    # Release the VideoCapture object  
    video.release()  


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an Image while maintaining aspect Ratio
    """

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def parseArgs():
    """
    Parses the command line arguments
    Params: 
    Return: argparse.Namespace
    """
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Set mode to Detect face in [image, video, webcam] : DEFAULT: webcam", type=str, default="webcam")
    parser.add_argument("-p", "--path", help="Path to image or video file requires --mode to be set to image or video", type=str)

    return parser.parse_args()


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
    # Run the main program
    main()
