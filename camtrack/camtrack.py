#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import click
import numpy as np
from cv2 import solvePnPRansac, solvePnP, findEssentialMat, RANSAC, findHomography, decomposeEssentialMat

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4
)


def init_first_camera_positions(intrinsic_mat, corner_storage):
    frame_count = len(corner_storage)

    best_pair = (-1, -1)
    zero_view_mat = eye3x4()
    best_view_mat = None
    best_triangulated_points = -1
    confidence = 0.9
    params = TriangulationParameters(max_reprojection_error=2, min_triangulation_angle_deg=1, min_depth=0.5)

    for i in range(frame_count):
        for j in range(i + 3, min(i + 30, frame_count), 3):
            correspondences = build_correspondences(corner_storage[i], corner_storage[j])
            if len(correspondences.ids) < 5:
                continue
            points_1, points_2 = correspondences.points_1, correspondences.points_2
            e_matrix, e_mask = findEssentialMat(points_1, points_2, intrinsic_mat, method=RANSAC, threshold=2,
                                                prob=confidence)
            h_matrix, h_mask = findHomography(points_1, points_2, method=RANSAC, ransacReprojThreshold=2,
                                              confidence=confidence)
            e_inliers, h_inliers = sum(e_mask.reshape(-1)), sum(h_mask.reshape(-1))
            if e_inliers / h_inliers < 0.1:
                continue
            outliers = np.delete(correspondences.ids, correspondences.ids[e_mask])
            correspondences = build_correspondences(corner_storage[i], corner_storage[j], outliers)
            R1, R2, t = decomposeEssentialMat(e_matrix)
            for rv in [R1, R2]:
                for tv in [-t, t]:
                    candidate_veiw_mat = np.hstack((rv, tv))
                    points, ids, _ = triangulate_correspondences(correspondences, zero_view_mat, candidate_veiw_mat,
                                                                 intrinsic_mat, params)
                    if len(points) > best_triangulated_points:
                        best_triangulated_points = len(points)
                        best_pair = (i, j)
                        best_view_mat = candidate_veiw_mat

    return (best_pair[0], zero_view_mat), (best_pair[1], best_view_mat)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(corner_storage)
    view_mats = [None] * frame_count

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_first_camera_positions(intrinsic_mat, corner_storage)
        id_1, view_mat_1 = known_view_1
        id_2, view_mat_2 = known_view_2
        view_mats[id_1], view_mats[id_2] = view_mat_1, view_mat_2
    else:
        id_1, pose_1 = known_view_1
        id_2, pose_2 = known_view_2
        view_mats[id_1] = pose_to_view_mat3x4(pose_1)
        view_mats[id_2] = pose_to_view_mat3x4(pose_2)
    params = TriangulationParameters(max_reprojection_error=2, min_triangulation_angle_deg=1, min_depth=0.5)
    correspondences = build_correspondences(corner_storage[id_1], corner_storage[id_2])
    points, ids, _ = triangulate_correspondences(correspondences, view_mats[id_1], view_mats[id_2], intrinsic_mat,
                                                 params)

    point_cloud_builder = PointCloudBuilder(ids.reshape(-1), points)

    while True:
        update = False
        for i in range(frame_count):
            if view_mats[i] is not None:
                continue
            corners = corner_storage[i]
            ids, ids1, ids2 = np.intersect1d(corners.ids, point_cloud_builder.ids, return_indices=True)
            points_2d, points_3d = corners.points[ids1], point_cloud_builder.points[ids2]
            retval, rvec, tvec, inliers = solvePnPRansac(points_3d, points_2d, intrinsic_mat, distCoeffs=None)
            if not retval:
                continue
            retval, rvec, tvec = solvePnP(points_3d[inliers], points_2d[inliers], intrinsic_mat, distCoeffs=None,
                                          rvec=rvec, tvec=tvec, useExtrinsicGuess=True)
            if not retval:
                continue
            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            update = True
            outliers = np.delete(ids, inliers)
            for j in range(i):
                if view_mats[j] is None:
                    continue
                correspondences = build_correspondences(corners, corner_storage[j], ids_to_remove=outliers)
                points, ids, _ = triangulate_correspondences(correspondences, view_mats[i], view_mats[j], intrinsic_mat,
                                                             params)
                point_cloud_builder.add_points(ids.reshape(-1), points)
            click.echo("Process frame %d/%d. %d 3D points found. inliners=%d"
                       % (i + 1, frame_count, len(point_cloud_builder.points), len(inliers)))
        if not update:
            break

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
