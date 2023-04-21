import cv2
import torch
from torch.backends import cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box

def detect_number_plates(video_path, output_video_path, in_img_size=640):
    # Load YOLOv5 model
    model = attempt_load("yolov5s.pt", map_location=torch.device("cpu"))
    model.eval()

    # Set up video reader and writer
    video_reader = cv2.VideoCapture(video_path)
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_reader.get(cv2.CAP_PROP_FPS),
        (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    # Set up variables for tracking number plates
    detected_number_plates = []
    prev_number_plate_coords = None
    prev_number_plate_frame_count = 0

    # Process each frame in the video
    while True:
        # Read a frame from the video
        ret, frame = video_reader.read()
        if not ret:
            break

        # Preprocess the frame
        img = model.preprocess(frame)
        img = torch.from_numpy(img).to(model.device)

        # Run inference
        outputs = model(img)

        # Postprocess the outputs
        for output in outputs:
            if output is not None and len(output) > 0:
                output[:, :4] = scale_coords(img.shape[2:], output[:, :4], frame.shape[:2]).round()

                for detection in output:
                    class_id = int(detection[5])
                    class_name = names[class_id]
                    if class_name == "license_plate":
                        confidence = detection[4]
                        bbox = detection[:4]
                        bbox = bbox.cpu().numpy().astype(int)
                        xmin, ymin, xmax, ymax = bbox
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                        # Store the detected number plate and its confidence interval
                        detected_number_plates.append((frame.shape[0], frame.shape[1], xmin, ymin, xmax, ymax, confidence))
        # Write the output frame to the output video
        video_writer.write(frame)

    # Release resources
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Save the detected number plates and their confidence intervals to a text file
    with open("detected_number_plates.txt", "w") as f:
        for number_plate in detected_number_plates:
            f.write(
                f"{number_plate[0]} {number_plate[1]} {number_plate[2]} {number_plate[3]} {number_plate[4]} {number_plate[5]} {number_plate[6]}\n"
            )

    return detected_number_plates

def scale_coords(img_shape, coords, img_org_shape):
    # Calculates the transformation necessary to convert the original image coordinates to the new (scaled) image coordinates.
    gain = torch.min(in_img_size / img_shape) # gain = old / new
    pad = (in_img_size - img_shape * gain) / 2 # wh padding
    pad = torch.Tensor([pad[0], pad[1], pad[0], pad[1]]).reshape(1, 4) # xyxy
    coords[:, :4] -= coords[:, :4].mean(0) # center
    coords[:, :4] *= gain
    coords[:, :4] += pad