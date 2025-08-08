import cv2
import numpy as np
import argparse


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hls_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # White color mask
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    # Yellow color mask
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered


def process_frame(frame):
    # Step 1: Color filtering
    filtered = hls_filter(frame)

    # Step 2: Grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Step 5: Region of interest
    height = frame.shape[0]
    width = frame.shape[1]
    roi_vertices = np.array(
        [
            [
                (int(0.1 * width), height),
                (int(0.4 * width), int(0.6 * height)),
                (int(0.6 * width), int(0.6 * height)),
                (int(0.9 * width), height),
            ]
        ],
        dtype=np.int32,
    )
    masked_edges = region_of_interest(edges, roi_vertices)

    # Step 6: Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300,
    )

    # Step 7: Draw lines on a blank image
    line_img = np.zeros_like(frame)
    draw_lines(line_img, lines)

    # Step 8: Combine original frame with line image
    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    return combo


def main(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = process_frame(frame)
        out.write(output_frame)

        # Optional: Show frame in a window
        cv2.imshow("Lane Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection on video.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    args = parser.parse_args()
    main(args.input, args.output)
