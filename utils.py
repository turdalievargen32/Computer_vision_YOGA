import numpy as np

def normalize_pose(pose):
    """
    Центрирует позу и масштабирует до длины 1 (L2-норма)
    """
    pose = np.array(pose)
    pose = pose - np.mean(pose, axis=0)  # Центрирование
    norm = np.linalg.norm(pose)
    if norm == 0:
        return pose
    return pose / norm

def compare_pose(user_pose, ref_pose):
    """
    Сравнивает две позы и возвращает процент совпадения (0-100)
    """
    user = normalize_pose(user_pose)
    ref = normalize_pose(ref_pose)
    
    # Убедимся, что длина совпадает
    if len(user) != len(ref):
        return 0

    # Евклидова дистанция
    distances = np.linalg.norm(user - ref, axis=1)
    mean_dist = np.mean(distances)

    # Преобразуем расстояние в "сходство"
    similarity = max(0, 1 - mean_dist)  # 1 - dist → [0..1]
    return round(similarity * 100, 2)
