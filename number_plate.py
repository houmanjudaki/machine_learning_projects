import cv2
import os
import tkinter as tk
from tkinter import filedialog
import shutil

# Path to the Haar Cascade model for license plate detection
plate_cascade_path = "./haarcascade_iranin_plate_number.xml"

# Output directory to save the extracted license plates
output_dir = "plates"
os.makedirs(output_dir, exist_ok=True)

def detect_license_plate(img):
    if img is None:
        print("Error: Unable to load image!")
        return

    # Load the Haar Cascade model
    plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
    if plate_cascade.empty():
        print("Error: Could not load the Haar Cascade model!")
        return

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the license plates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    min_area = 500
    count = 0

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw a rectangle around the license plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the license plate region
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

            # Save the extracted license plate image
            cv2.imwrite(os.path.join(output_dir, f"scanned_img_{count}.jpg"), img_roi)
            count += 1



def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;")])
    if file_path:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(os.getcwd(), file_name)
        shutil.copy(file_path, destination_path)
        label.config(text=f"File saved to: {destination_path}")
        img = cv2.imread(destination_path)
        detect_license_plate(img)

def use_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam")
            break

        # Detect license plates continuously
        detect_license_plate(frame)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Image Selector")

# Set the window size (widthxheight)
root.geometry("400x200")

# Create a button to select the file
button_select_image = tk.Button(root, text="Select Image", command=select_file)
button_select_image.pack(pady=10)

# Create a button to use the webcam
button_use_webcam = tk.Button(root, text="Use Webcam", command=use_webcam)
button_use_webcam.pack(pady=10)

# Create a label to display the result
label = tk.Label(root, text="برای بستن حالت وب کم q ")
label.pack(pady=20)

# Run the application
root.mainloop()
