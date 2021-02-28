#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


from numba import jit


@jit
def id_closer_than(array, value, distance):
    d = distance
    for x in array:
        if abs(x[0] - value[0]) < d and abs(x[1] - value[1]) < d:
            return x[2]
    return array.shape[0]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    maxCorners = (image_0.shape[0] * image_0.shape[1]) // 2000
    qualityLevel = 0.001
    minDistance = 20
    blockSize = 9
    winSize = (20, 20)

    corner_points = cv2.goodFeaturesToTrack(image_0, maxCorners, qualityLevel, minDistance, blockSize=blockSize,
                                            useHarrisDetector=False).reshape((-1, 2))
    max_id = len(corner_points)
    ids = np.arange(0, max_id)
    corners = FrameCorners(ids, corner_points, np.full(corner_points.shape, blockSize))
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
            np.asarray(image_0 * 256, np.uint8),
            np.asarray(image_1 * 256, np.uint8),
            corner_points.astype(np.float32), None, winSize=winSize)
        ptid = np.c_[nextPts, ids][status.reshape(-1) == 1]
        b = np.fromiter((id_closer_than(ptid, xi, minDistance) >= xi[2] for xi in ptid), bool, count=len(ptid))
        ptid = ptid[b]
        corner_points, ids = np.hsplit(ptid, [2])
        ids = ids.reshape(-1).astype(int)

        new_corner_points = cv2.goodFeaturesToTrack(image_1, maxCorners, qualityLevel, minDistance, blockSize=blockSize,
                                                    useHarrisDetector=False).reshape((-1, 2))
        b = np.fromiter((id_closer_than(ptid, xi, minDistance) == ptid.shape[0] for xi in new_corner_points), bool,
                        count=len(new_corner_points))
        new_corner_points = new_corner_points[b][:maxCorners - corner_points.shape[0]]
        new_ids = np.arange(max_id, max_id + new_corner_points.shape[0])
        max_id += new_corner_points.shape[0]
        corner_points = np.concatenate((corner_points, new_corner_points))
        ids = np.concatenate((ids, new_ids))

        corners = FrameCorners(ids, corner_points, np.full(corner_points.shape, blockSize))
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
