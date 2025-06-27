import numpy as np

def normalize_pose(pose):
    """
    Центрирует позу и масштабирует до длины 1 (L2-норма)
    """
    pose = np.array(pose)
    pose = pose - np.mean(pose, axis=0)  #
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
    
    if len(user) != len(ref):
        return 0
    distances = np.linalg.norm(user - ref, axis=1)
    mean_dist = np.mean(distances)

    
    similarity = max(0, 1 - mean_dist)  
    return round(similarity * 100, 2)
