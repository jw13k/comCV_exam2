import cv2
import numpy as np
import torch
from mss import mss
import mediapipe as mp

# YOLOv11 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # 스크린 캡처 객체 생성
    sct = mss()
    monitor = sct.monitors[1]  # 첫 번째 모니터 선택 (필요 시 [2], [3]으로 변경)

    while True:
        # 화면 캡처
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 객체 감지
        results = model(frame)

        # 결과를 프레임에 그리기
        for *box, conf, cls in results.xyxy[0]:
            if model.names[int(cls)] == 'person' and conf >= 0.5:
                # 사람 영역 자르기
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                person_image = frame[y1:y2, x1:x2]

                # OpenPose로 자세 추정 (해상도 축소)
                person_image_resized = cv2.resize(person_image, (256, 256))
                results_pose = pose.process(cv2.cvtColor(person_image_resized, cv2.COLOR_BGR2RGB))

                # 결과 시각화
                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(person_image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    frame[y1:y2, x1:x2] = person_image

        # 프레임 크기 조정 (작게)
        small_frame = cv2.resize(frame, (800, 450))

        # 프레임 출력
        cv2.imshow('Screen Intrusion Detection', small_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cv2.destroyAllWindows()