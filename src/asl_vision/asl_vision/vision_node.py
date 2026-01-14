import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ASLVisionNode(Node):
    def __init__(self):
        super().__init__('asl_vision_node')

        # subscribe to our camera node
        self.subscription = self.create_subscription(
                Image,
                'image_raw',
                self.listener_callback,
                10)

        self.bridge = CvBridge()

        # init mediapipe landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1,
                                               min_hand_detection_confidence=0.5,
                                               min_hand_presence_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.frame_count = 0
    
    def listener_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 2 != 0: # only process every other frame (optimization)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(frame, (320, 240)) # downscale (optimization)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detection_result = self.detector.detect(mp_image)

            prediction = "None"
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                prediction = self.classify_gesture(hand_landmarks)

                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(frame, f'Sign: {prediction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ASL Recognition", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"error: {str(e)}")

    def classify_gesture(self, landmarks):
        itip = landmarks[8]
        ipip = landmarks[6]
        mtip = landmarks[12]
        mpip = landmarks[10]

        index_up = itip.y < ipip.y
        middle_up = mtip.y < mpip.y

        # check for "V" or "L"
        if index_up and middle_up:
            return "V"
        elif index_up and not middle_up:
            return "L"
        else:
            return "unknown"

def main(args=None):
    rclpy.init(args=args)
    node = ASLVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
