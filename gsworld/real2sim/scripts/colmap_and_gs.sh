#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash colmap_and_gs.sh --data_dir <path> [options]

Runs:
  1) COLMAP SfM on <data_dir>/images
  2) Metric rescaling using an ArUco marker
  3) 3D Gaussian Splatting (3DGS) training

Required:
  --data_dir PATH        Dataset directory containing an images/ folder

Options:
  --gpu ID               CUDA device id (default: 0)
  --aruco_size METERS    ArUco marker side length in meters (default: 0.100)
  --colmap CMD           COLMAP executable (default: colmap)
  --camera MODEL         COLMAP camera model (default: PINHOLE)
  --iterations N         3DGS iterations (default: 30000)
  --model_path PATH      3DGS output folder (default: submodules/gaussian-splatting/output/<run_id>)
  --export_ply PATH      If set, copy the final point_cloud.ply to this path (relative paths are repo-root relative)
  --skip_sfm             Skip COLMAP SfM
  --skip_aruco           Skip ArUco rescale
  --skip_train           Skip 3DGS training
  -h, --help             Show help

Example:
  bash colmap_and_gs.sh --data_dir data/my_capture --gpu 0 --aruco_size 0.100 \
    --export_ply assets/galaxea_r1_assets/my_scene.ply
EOF
}

DATA_DIR=""
GPU="0"
ARUCO_SIZE="0.100"
COLMAP_COMMAND="colmap"
CAMERA_MODEL="PINHOLE"
ITERATIONS="30000"
MODEL_PATH=""
EXPORT_PLY=""
SKIP_SFM="0"
SKIP_ARUCO="0"
SKIP_TRAIN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --aruco_size) ARUCO_SIZE="$2"; shift 2 ;;
    --colmap) COLMAP_COMMAND="$2"; shift 2 ;;
    --camera) CAMERA_MODEL="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --export_ply) EXPORT_PLY="$2"; shift 2 ;;
    --skip_sfm) SKIP_SFM="1"; shift ;;
    --skip_aruco) SKIP_ARUCO="1"; shift ;;
    --skip_train) SKIP_TRAIN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${DATA_DIR}" ]]; then
  echo "Error: --data_dir is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Error: --data_dir does not exist: ${DATA_DIR}" >&2
  exit 1
fi
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"

if [[ ! -d "${DATA_DIR}/images" ]]; then
  echo "Error: expected an images/ folder under --data_dir: ${DATA_DIR}/images" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  RUN_ID="$(basename "${DATA_DIR}")_$(date +%Y%m%d_%H%M%S)"
  MODEL_PATH="${REPO_ROOT}/submodules/gaussian-splatting/output/${RUN_ID}"
elif [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
fi

if [[ -n "${EXPORT_PLY}" && "${EXPORT_PLY}" != /* ]]; then
  EXPORT_PLY="${REPO_ROOT}/${EXPORT_PLY}"
fi

echo "[colmap_and_gs] Repo root: ${REPO_ROOT}"
echo "[colmap_and_gs] Data dir:   ${DATA_DIR}"
echo "[colmap_and_gs] Model path: ${MODEL_PATH}"

if [[ "${SKIP_SFM}" -eq 0 ]]; then
  echo "[colmap_and_gs] (1/3) Running COLMAP SfM..."
  (cd "${SCRIPT_DIR}" && python sfm.py --source_path "${DATA_DIR}" --colmap-command "${COLMAP_COMMAND}" --camera "${CAMERA_MODEL}" -v)
else
  echo "[colmap_and_gs] (1/3) Skipping COLMAP SfM (--skip_sfm)."
fi

if [[ "${SKIP_ARUCO}" -eq 0 ]]; then
  echo "[colmap_and_gs] (2/3) Applying ArUco metric rescale..."
  (cd "${SCRIPT_DIR}" && python aruco_rescale.py --source_path "${DATA_DIR}" --aruco-size "${ARUCO_SIZE}")
else
  echo "[colmap_and_gs] (2/3) Skipping ArUco rescale (--skip_aruco)."
fi

if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  echo "[colmap_and_gs] (3/3) Training 3D Gaussian Splatting..."
  GS_DIR="${REPO_ROOT}/submodules/gaussian-splatting"
  if [[ ! -d "${GS_DIR}" ]]; then
    echo "Error: gaussian-splatting submodule not found: ${GS_DIR}" >&2
    exit 1
  fi
  (cd "${GS_DIR}" && CUDA_VISIBLE_DEVICES="${GPU}" python train.py -s "${DATA_DIR}" -m "${MODEL_PATH}" --disable_viewer --iterations "${ITERATIONS}")
else
  echo "[colmap_and_gs] (3/3) Skipping 3DGS training (--skip_train)."
fi

if [[ -n "${EXPORT_PLY}" ]]; then
  echo "[colmap_and_gs] Exporting final point cloud to: ${EXPORT_PLY}"
  mkdir -p "$(dirname "${EXPORT_PLY}")"

  POINT_CLOUD_DIR="${MODEL_PATH}/point_cloud"
  if [[ ! -d "${POINT_CLOUD_DIR}" ]]; then
    echo "Error: point_cloud folder not found: ${POINT_CLOUD_DIR}" >&2
    exit 1
  fi

  PLY_SRC=""
  if [[ -f "${POINT_CLOUD_DIR}/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
    PLY_SRC="${POINT_CLOUD_DIR}/iteration_${ITERATIONS}/point_cloud.ply"
  else
    LATEST_ITER_DIR="$(ls -1d "${POINT_CLOUD_DIR}/iteration_"* 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_ITER_DIR}" && -f "${LATEST_ITER_DIR}/point_cloud.ply" ]]; then
      PLY_SRC="${LATEST_ITER_DIR}/point_cloud.ply"
    fi
  fi

  if [[ -z "${PLY_SRC}" ]]; then
    echo "Error: could not find point_cloud.ply under ${POINT_CLOUD_DIR}" >&2
    exit 1
  fi

  cp "${PLY_SRC}" "${EXPORT_PLY}"
  echo "[colmap_and_gs] Exported: ${EXPORT_PLY}"
fi

echo "[colmap_and_gs] Done."
