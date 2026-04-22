import cv2
import os
import time

def main():
    output_folder = "captured_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open /dev/video2")
        return

    ## Set resolution to 960x540 and frame rate to 60Hz
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    ## Reduce buffer to 1 to get the freshest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            file_name = f"frame_{frame_count:05d}.jpg"
            file_path = os.path.join(output_folder, file_name)

            # cv2.imwrite(file_path, frame)

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()