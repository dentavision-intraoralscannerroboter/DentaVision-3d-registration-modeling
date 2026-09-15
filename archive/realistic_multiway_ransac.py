from pathlib import Path
import copy

import numpy as np
import open3d as o3d
import pandas as pd

from realistic_registration_benchmark import (
    load_cloud,
    preprocess,
    ransac_fpfh_icp,
    evaluate,
    VOXEL_SIZE,
)


BASE_DIR = Path("data/realistic_jaw")
OUTPUT_BASE = Path("output/realistic_multiway_ransac")

JAW_DIRS = {
    "upper": BASE_DIR / "upper_test",
    "lower": BASE_DIR / "lower_test",
}


def rotation_error_deg(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    r_est = estimated[:3, :3]
    r_gt = ground_truth[:3, :3]

    delta = r_est @ r_gt.T

    value = (np.trace(delta) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def translation_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            estimated[:3, 3]
            - ground_truth[:3, 3]
        )
    )


def process_jaw(
    jaw_name: str,
    jaw_dir: Path,
):
    print()
    print("=" * 80)
    print(f"{jaw_name.upper()} JAW – MULTIWAY RANSAC")
    print("=" * 80)

    output_dir = OUTPUT_BASE / jaw_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    acquisition_transforms = np.load(
        jaw_dir / "acquisition_transforms.npy"
    )

    ground_truth_poses = np.load(
        jaw_dir / "ground_truth_poses.npy"
    )

    original_clouds = []
    prepared = {}

    for index in range(4):
        cloud = load_cloud(
            jaw_dir / f"fragment_{index:02d}.ply"
        )

        original_clouds.append(cloud)

        downsampled, fpfh = preprocess(cloud)

        prepared[index] = {
            "cloud": downsampled,
            "fpfh": fpfh,
        }

    pairwise_transforms = []

    # ---------------------------------------------------------
    # Pairwise Odometry: 0->1, 1->2, 2->3
    # ---------------------------------------------------------

    for source_index in range(3):
        target_index = source_index + 1

        print()
        print(
            f"Registriere "
            f"{source_index} -> {target_index}"
        )

        source = prepared[source_index]["cloud"]
        target = prepared[target_index]["cloud"]

        result = ransac_fpfh_icp(
            source,
            target,
            prepared[source_index]["fpfh"],
            prepared[target_index]["fpfh"],
        )

        transform = result.transformation

        pairwise_transforms.append(transform)

        # Ground Truth:
        # x_i = A_i @ x_world
        #
        # i -> j:
        # T_gt = A_j @ inv(A_i)

        gt_pair = (
            acquisition_transforms[target_index]
            @ np.linalg.inv(
                acquisition_transforms[source_index]
            )
        )

        metrics = evaluate(
            source,
            target,
            transform,
            gt_pair,
        )

        print(
            f"fitness={metrics['fitness']:.4f}, "
            f"rmse={metrics['inlier_rmse']:.4f}, "
            f"t_err={metrics['translation_error']:.6f}, "
            f"r_err={metrics['rotation_error_deg']:.6f}°"
        )

    # ---------------------------------------------------------
    # Globale Posen aufbauen
    #
    # Pairwise:
    # T_i_j führt Fragment i -> Fragment j
    #
    # Gesucht:
    # Pose_i führt Fragment i -> Fragment 0
    # ---------------------------------------------------------

    poses = [np.eye(4)]

    for transform in pairwise_transforms:

        next_pose = (
            poses[-1]
            @ np.linalg.inv(transform)
        )

        poses.append(next_pose)

    rows = []

    combined = o3d.geometry.PointCloud()

    print()
    print("-" * 80)
    print("GLOBALE POSEFEHLER")
    print("-" * 80)

    for index, (
        cloud,
        estimated_pose,
    ) in enumerate(
        zip(
            original_clouds,
            poses,
        )
    ):

        gt_pose = ground_truth_poses[index]

        t_error = translation_error(
            estimated_pose,
            gt_pose,
        )

        r_error = rotation_error_deg(
            estimated_pose,
            gt_pose,
        )

        print(
            f"Fragment {index}: "
            f"t_err={t_error:.6f} mm, "
            f"r_err={r_error:.6f}°"
        )

        rows.append(
            {
                "jaw": jaw_name,
                "fragment": index,
                "translation_error_mm": t_error,
                "rotation_error_deg": r_error,
            }
        )

        registered = copy.deepcopy(cloud)

        registered.transform(
            estimated_pose
        )

        registered_path = (
            output_dir
            / f"registered_fragment_{index:02d}.ply"
        )

        o3d.io.write_point_cloud(
            str(registered_path),
            registered,
        )

        combined += registered

    # ---------------------------------------------------------
    # Zusammengeführte Punktwolke
    # ---------------------------------------------------------

    combined = combined.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    combined_path = (
        output_dir
        / "combined_cloud.ply"
    )

    o3d.io.write_point_cloud(
        str(combined_path),
        combined,
    )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_dir / "pose_errors.csv",
        index=False,
    )

    np.save(
        output_dir / "estimated_poses.npy",
        np.stack(poses),
    )

    print()
    print(
        f"Combined cloud: "
        f"{len(combined.points):,} Punkte"
    )

    print(
        f"Gespeichert: {combined_path}"
    )

    return rows


def main():

    OUTPUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for jaw_name, jaw_dir in JAW_DIRS.items():

        all_rows.extend(
            process_jaw(
                jaw_name,
                jaw_dir,
            )
        )

    summary = pd.DataFrame(all_rows)

    summary.to_csv(
        OUTPUT_BASE / "all_pose_errors.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("MULTIWAY-GESAMTERGEBNIS")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()