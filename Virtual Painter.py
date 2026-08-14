import cv2
import mediapipe as mp
import numpy as np
CAMERA_ID = 0
BRUSH_SIZE = 30
ERASER_SIZE = 90
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera 0 could not be opened.")
    print("Try changing CAMERA_ID = 0 to CAMERA_ID = 1")
    exit()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
canvas = None
current_color = (0, 0, 255)   # Red
previous_x = 0
previous_y = 0
colors = [
    (0, 0, 255),       # RED
    (0, 255, 0),       # GREEN
    (255, 0, 0),       # BLUE
    (0, 255, 255),     # YELLOW
    (255, 0, 255),     # MAGENTA
    (255, 255, 0),     # CYAN
    (128,0,128),       # PUPLE
    (128,128,128),     #GRAY
    (0,165,165),       # ORANGE
    (128,0,0),         # NAVY
    (0,0,128),        #MAROON
    (255,255,255),    #WHITE 
    (0, 0, 0)          # BLACK
]
COLOR_BOX_WIDTH = 100
def fingers_up(hand):

    landmarks = hand.landmark

    fingers = []

    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    
    if landmarks[8].y < landmarks[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    
    if landmarks[12].y < landmarks[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    
    if landmarks[16].y < landmarks[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    
    if landmarks[20].y < landmarks[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers
def draw_color_bar(image):

    cv2.rectangle(
        image,
        (0, 0),
        (1280, 80),
        (40, 40, 40),
        -1
    )
   
    for i, color in enumerate(colors):

        x1 = i * COLOR_BOX_WIDTH + 10
        x2 = i * COLOR_BOX_WIDTH + 90

        cv2.rectangle(
            image,
            (x1, 10),
            (x2, 70),
            color,
            -1
        )

        if color == current_color:

            cv2.rectangle(
                image,
                (x1 - 3, 7),
                (x2 + 3, 73),
                (255, 255, 255),
                3
            )
WIDTH = 1600
HEIGHT = 900

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

cv2.namedWindow("AI Virtual Painter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Virtual Painter", WIDTH, HEIGHT)


while True:

    
    success, frame = cap.read()

    if not success:

        print("Could not read camera frame.")
        break

    
    frame = cv2.flip(frame, 1)

    
    if canvas is None:

        canvas = np.zeros_like(frame)

    

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = detector.process(rgb_frame)


    draw_color_bar(frame)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        
        fingers = fingers_up(hand)

        
        index_tip = hand.landmark[8]

        x = int(
            index_tip.x * frame.shape[1]
        )

        y = int(
            index_tip.y * frame.shape[0]
        )

        
        if fingers[1] == 1 and fingers[2] == 1:

            if y < 80:

                color_number = x // COLOR_BOX_WIDTH

                if 0 <= color_number < len(colors):

                    current_color = colors[color_number]

                previous_x = None
                previous_y = None

            

            else:

                cv2.circle(
                    canvas,
                    (x, y),
                    ERASER_SIZE,
                    (0, 0, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    (x, y),
                    ERASER_SIZE,
                    (255, 255, 255),
                    2
                )

                previous_x = None
                previous_y = None

        elif fingers[1] == 1 and fingers[2] == 0:

            if y > 80:

                cv2.circle(
                    frame,
                    (x, y),
                    BRUSH_SIZE + 2,
                    current_color,
                    2
                )

                if previous_x is not None:

                    cv2.line(
                        canvas,
                        (previous_x, previous_y),
                        (x, y),
                        current_color,
                        BRUSH_SIZE,
                        cv2.LINE_AA
                    )

                previous_x = x
                previous_y = y

            else:

                previous_x = None
                previous_y = None

        else:

            previous_x = None
            previous_y = None

    else:

        previous_x = None
        previous_y = None

    
    
    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray_canvas,
        10,
        255,
        cv2.THRESH_BINARY
    )

    camera_background = cv2.bitwise_and(
        frame,
        frame,
        mask=cv2.bitwise_not(mask)
    )

    drawing = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    output = cv2.add(
        camera_background,
        drawing
    )

    draw_color_bar(output)

    cv2.putText(
        output,
        "DRAW",
        (770, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "ERASER",
        (770, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Virtual Painter",
        output
    )

    
    key = cv2.waitKey(1) & 0xFF

    
    if key == ord("q"):

        break


cap.release()

detector.close()

cv2.destroyAllWindows()