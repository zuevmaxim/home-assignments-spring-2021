#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import click
import numpy as np
from cv2 import solvePnPRansac

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
    rodrigues_and_translation_to_view_mat3x4
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    params = TriangulationParameters(max_reprojection_error=1, min_triangulation_angle_deg=1, min_depth=0.5)

    id_1, pose_1 = known_view_1
    id_2, pose_2 = known_view_2
    view_mats[id_1] = pose_to_view_mat3x4(pose_1)
    view_mats[id_2] = pose_to_view_mat3x4(pose_2)
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
            ids = np.intersect1d(corners.ids, point_cloud_builder.ids)
            ids = set(ids)
            points_2d = np.array([p for z, p in zip(corners.ids.reshape(-1), corners.points) if z in ids], )
            points_3d = np.array([p for z, p in zip(point_cloud_builder.ids.reshape(-1), point_cloud_builder.points)
                                  if z in ids])
            retval, rvec, tvec, inliers = solvePnPRansac(points_3d, points_2d, intrinsic_mat, distCoeffs=None)
            if not retval:
                continue
            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            update = True
            for j in range(i):
                if view_mats[j] is None:
                    continue
                correspondences = build_correspondences(corners, corner_storage[j])
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
