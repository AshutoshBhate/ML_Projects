import cv2
import os

# Prompt for the person's name to create a folder for saving images
person_name = input("Enter the name of the person: ").strip().lower()
save_dir = os.path.join("Photos", person_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
img_count = 0

print("Starting image capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    cv2.imshow("Webcam", frame)

    # Save the captured frame into the person's folder
    img_count += 1
    img_path = os.path.join(save_dir, f"image_{img_count}.png")
    cv2.imwrite(img_path, frame)
    print(f"Saved {img_path}")

    # Wait for about 2 seconds before capturing the next image
    for _ in range(20):  # 20 iterations * 100ms = 2 seconds
        if cv2.waitKey(100) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
