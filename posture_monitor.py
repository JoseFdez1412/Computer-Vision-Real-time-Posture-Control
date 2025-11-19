import cv2
import time
import math
import numpy as np
from plyer import notification
from ultralytics import YOLO

# ==============================
# 1) Parámetros de configuración
# ==============================

BAD_POSTURE_THRESHOLD_SECONDS = 3 * 60  # 3 minutos
CAMERA_INDEX = 0  # Índice de la cámara
TORSO_ANGLE_THRESHOLD_DEG = 25  # Umbral de inclinación de tronco en grados
HEAD_FORWARD_THRESHOLD_REL = 0.07  # Umbral relativo para "cabeza adelantada"


# ==============================
# 2) Carga del modelo de pose (YOLOv8)
# ==============================

# Usamos el modelo más pequeño para ir más ligero: yolov8n-pose
# La primera vez que lo ejecutes, descargará el .pt automáticamente.
model = YOLO("yolov8n-pose.pt")


# ==============================
# 3) Funciones auxiliares de geometría
# ==============================

def angle_with_vertical(p1, p2):
    """
    Calcula el ángulo entre el vector p2->p1 y el eje vertical (0, -1).
    p1, p2: np.array([x, y])
    Devuelve el ángulo en grados.
    """
    v = p1 - p2  # vector desde p2 a p1 (por ejemplo, cadera -> hombro)
    if np.linalg.norm(v) < 1e-6:
        return 0.0

    v_norm = v / np.linalg.norm(v)
    vertical = np.array([0.0, -1.0])  # apuntando hacia arriba en la imagen
    dot = np.clip(np.dot(v_norm, vertical), -1.0, 1.0)
    angle_rad = math.acos(dot)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


# ==============================
# 4) Estimación de pose con YOLOv8
# ==============================

def estimate_pose(frame):
    """
    Usa YOLOv8 Pose para obtener los keypoints de la persona principal.
    Devuelve:
        keypoints: np.array shape (17, 2) en píxeles (x, y),
                   o None si no hay detección.
    """
    results = model(frame, verbose=False)

    if len(results) == 0:
        return None

    r = results[0]

    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return None

    # Tomamos la primera persona detectada
    kp_xy = r.keypoints.xy[0].cpu().numpy()  # shape (17, 2)
    return kp_xy


# ==============================
# 5) Lógica para determinar mala postura
# ==============================

def is_bad_posture_from_keypoints(keypoints, frame_shape):
    """
    Decide si la postura es mala basándose en:
        - Ángulo del tronco (caderas -> hombros).
        - Posición adelantada de la cabeza respecto al tronco.
    """
    h, w, _ = frame_shape

    # Índices COCO típicos:
    # 5: left_shoulder, 6: right_shoulder
    # 11: left_hip, 12: right_hip
    # 0: nose, 3: left_ear, 4: right_ear
    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
    except IndexError:
        return False, {"reason": "missing_keypoints"}

    # Centro de hombros y caderas
    mid_shoulders = (left_shoulder + right_shoulder) / 2.0
    mid_hips = (left_hip + right_hip) / 2.0

    # 1) Ángulo del tronco
    torso_angle = angle_with_vertical(mid_shoulders, mid_hips)

    # 2) Cabeza adelantada
    head_points = []
    for idx in [0, 3, 4]:
        if idx < keypoints.shape[0]:
            head_points.append(keypoints[idx])
    if len(head_points) > 0:
        head_center = np.mean(head_points, axis=0)
    else:
        head_center = mid_shoulders  # fallback

    head_offset_x = head_center[0] - mid_shoulders[0]
    head_offset_rel = head_offset_x / w

    bad_torso = torso_angle > TORSO_ANGLE_THRESHOLD_DEG
    bad_head = head_offset_rel > HEAD_FORWARD_THRESHOLD_REL

    bad_posture = bad_torso or bad_head

    metrics = {
        "torso_angle_deg": float(torso_angle),
        "head_offset_rel": float(head_offset_rel),
        "bad_torso": bool(bad_torso),
        "bad_head": bool(bad_head)
    }

    return bad_posture, metrics


# ==============================
# 6) Notificación (popup)
# ==============================

def show_posture_notification():
    notification.notify(
        title="Posture Monitor",
        message="Parece que llevas un rato con mala postura. ¡Siéntate derecho! 🙂",
        timeout=10
    )


# ==============================
# 7) Bucle principal
# ==============================

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    bad_posture_start_time = None
    last_notification_time = 0

    print("Posture monitor corriendo. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        # Estimación de pose
        keypoints = estimate_pose(frame)

        if keypoints is not None:
            bad_posture, metrics = is_bad_posture_from_keypoints(keypoints, frame.shape)
        else:
            bad_posture, metrics = False, {"reason": "no_person"}

        # Lógica de tiempo + notificación
        current_time = time.time()

        if bad_posture:
            if bad_posture_start_time is None:
                bad_posture_start_time = current_time

            elapsed_bad = current_time - bad_posture_start_time
            print(f"bad_posture=True, elapsed_bad={elapsed_bad:.2f} s")

            if (elapsed_bad > BAD_POSTURE_THRESHOLD_SECONDS and
                    current_time - last_notification_time > BAD_POSTURE_THRESHOLD_SECONDS / 2):
                print(">>> Disparando notificación de mala postura")
                show_posture_notification()
                last_notification_time = current_time
        else:
            if bad_posture_start_time is not None:
                print("Postura corregida, reseteando contador")
            bad_posture_start_time = None

        # Visualización
        text = f"Bad posture: {bad_posture}"
        cv2.putText(frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if bad_posture else (0, 255, 0),
                    2)

        if "torso_angle_deg" in metrics:
            cv2.putText(frame,
                        f"Torso angle: {metrics['torso_angle_deg']:.1f} deg",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)
        if "head_offset_rel" in metrics:
            cv2.putText(frame,
                        f"Head offset: {metrics['head_offset_rel']:.3f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        # Dibujar keypoints clave
        if keypoints is not None:
            for idx in [5, 6, 11, 12, 0, 3, 4]:
                if idx < keypoints.shape[0]:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        cv2.imshow("Posture Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
