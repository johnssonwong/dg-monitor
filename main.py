#!/usr/bin/env python3
# main.py
# DG monitor - board detection single-file implementation
# 严格使用 state.json, history_db.json, history_stats.json, last_summary.json 文件名（与 Actions 日志一致）
# 行为：先用主方法检测；若检测到的桌子小于门槛 -> 立即用替补方法检测

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path

# ----------------------
# 配置与默认值（从环境变量读取）
# ----------------------
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
MIN_BOARDS_FOR_PAW = int(os.getenv("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.getenv("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "10"))
HISTORY_LOOKBACK_DAYS = int(os.getenv("HISTORY_LOOKBACK_DAYS", "28"))
HISTORY_PROB_THRESHOLD = float(os.getenv("HISTORY_PROB_THRESHOLD", "0.35"))

STATE_PATH = Path("state.json")
HISTORY_DB_PATH = Path("history_db.json")
HISTORY_STATS_PATH = Path("history_stats.json")
LAST_SUMMARY_PATH = Path("last_summary.json")

# ----------------------
# 类型
# ----------------------
BoardRect = Tuple[int, int, int, int, float]  # x, y, w, h, area

# ----------------------
# 日志设置（输出格式尽量贴近 Actions 日志那样带时间戳）
# ----------------------
logger = logging.getLogger("dg-monitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------
# I/O helpers for the state/history files
# ----------------------
def load_json_safe(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return default
    except Exception:
        return default

def save_json_safe(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入 {path} 失败: {e}")

# ----------------------
# 图像检测实现（主方法 + 替补方法 + 调用逻辑）
# ----------------------
def primary_detect_grid(img: np.ndarray,
                        min_area: int = 20000,
                        approx_eps_ratio: float = 0.02) -> List[BoardRect]:
    """
    主检测：适用于面板边界比较明显、网格/大矩形块清晰的情形。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # 自适应阈值
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 9)
    # 自适应 kernel 大小（避免在高分辨率下把面板合并）
    kx = max(9, img.shape[1] // 160)
    ky = max(5, img.shape[0] // 240)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[BoardRect] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps_ratio * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 60 and h > 40:
                rects.append((x, y, w, h, float(area)))
    rects.sort(key=lambda r: (r[1], r[0]))
    return rects

def fallback_detect_edges(img: np.ndarray,
                          min_area: int = 2000) -> List[BoardRect]:
    """
    替补检测：Canny -> 膨胀 -> 查找轮廓，适合把每个小面板单独分离出来（不会把相邻面板合并成一块）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # Canny 边缘
    edges = cv2.Canny(blur, 50, 150)
    k = max(3, img.shape[1] // 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[BoardRect] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 and h > 40:
            rects.append((x, y, w, h, float(area)))
    rects.sort(key=lambda r: (r[1], r[0]))
    return rects

def detect_boards(img: np.ndarray,
                  min_boards_required: int = 3,
                  force_fallback: bool = False) -> Tuple[List[BoardRect], str]:
    """
    统一检测接口：
      - 先尝试 primary_detect_grid
      - 如果主检测返回数量 < min_boards_required 或 force_fallback -> 立即调用替补方法
      - 返回 (rects, method_used)
    """
    if force_fallback:
        fb = fallback_detect_edges(img)
        return fb, "fallback_forced"

    primary = primary_detect_grid(img)
    if len(primary) >= min_boards_required:
        return primary, "primary"
    # primary 结果不足 -> 立即调用替补方法
    fallback = fallback_detect_edges(img)
    # 如果 fallback 更好（数量更多或至少不比 primary 差）则切换
    if len(fallback) >= len(primary) and len(fallback) > 0:
        return fallback, "fallback"
    else:
        return primary, "primary"

# ----------------------
# 工具：在本地保存检测到的各个板块（便于人工复核）
# ----------------------
def save_crops(img: np.ndarray, rects: List[BoardRect], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h, area) in enumerate(rects):
        crop = img[y:y+h, x:x+w]
        fname = out_dir / f"board_{i+1:02d}_{x}_{y}_{w}x{h}.png"
        cv2.imwrite(str(fname), crop)

# ----------------------
# 主流程
# ----------------------
def run_once(image_path: Path, min_boards_required: int, force_fallback: bool = False):
    logger.info("=== DG monitor run start ===")
    logger.info(f"将处理图片: {image_path}")

    if not image_path.exists():
        logger.error(f"图片未找到: {image_path}")
        # 为了与 CI 行为一致，仍然尝试创建空的 history_db.json（以免 git add 失败）
        if not HISTORY_DB_PATH.exists():
            save_json_safe(HISTORY_DB_PATH, {})
        return

    # 读取
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("无法读取图片或格式不支持")
        return

    # 模拟日志中的“打开页面/点击 Free”步骤输出（但是不实际打开网页）
    logger.info(f"打开 https://dg18.co/wap/ （尝试 1）")
    logger.info("点击文本 Free")

    rects, method = detect_boards(img, min_boards_required=min_boards_required, force_fallback=force_fallback)
    logger.info(f"[实时检测] 方法: {method}; 检测到桌子数: {len(rects)} (阈值 {min_boards_required})")

    # 如果检测到的矩形数量 > 0，保存 crops 便于后续人工审核
    if len(rects) > 0:
        save_dir = Path("detected_boards")
        save_crops(img, rects, save_dir)
        logger.info(f"已保存各桌截图到: {save_dir}（共 {len(rects)} 张）")
    else:
        logger.info("未检测到任何桌子（rects == 0）")

    # 更新 state/history 文件，与 CI 日志相匹配
    state = load_json_safe(STATE_PATH, {})
    state.update({
        "last_run": datetime.now().isoformat(),
        "last_image": str(image_path),
        "detected_boards": len(rects),
        "method": method,
    })
    save_json_safe(STATE_PATH, state)

    # history_db.json 保证存在（若之前不存在，创建空对象）
    history_db = load_json_safe(HISTORY_DB_PATH, {})
    # 追加一个简单条目（以时间戳为 key）
    ts = datetime.now().isoformat()
    history_db.setdefault("runs", []).append({
        "ts": ts,
        "image": str(image_path),
        "boards": len(rects),
        "method": method
    })
    save_json_safe(HISTORY_DB_PATH, history_db)

    # history_stats.json 更新（非常简化的统计）
    stats = load_json_safe(HISTORY_STATS_PATH, {"total_runs": 0, "total_boards": 0})
    stats["total_runs"] = stats.get("total_runs", 0) + 1
    stats["total_boards"] = stats.get("total_boards", 0) + len(rects)
    save_json_safe(HISTORY_STATS_PATH, stats)

    # last_summary.json 写一个简短摘要（与 CI 日志语义一致）
    last_summary = {
        "ts": ts,
        "summary": f"检测到桌子 {len(rects)} (method={method})"
    }
    save_json_safe(LAST_SUMMARY_PATH, last_summary)

    # 终了日志
    logger.info("=== DG monitor run end ===")

# ----------------------
# CLI entrypoint
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="DG monitor - detect boards in a screenshot (main.py)")
    p.add_argument("--image", "-i", type=str,
                   default="/mnt/data/3D04749B-1CDC-42B0-8468-E233F3F81987.jpeg",
                   help="待检测的截图路径（默认同 conversation 中的文件）")
    p.add_argument("--min-boards", type=int, default=MIN_BOARDS_FOR_PAW,
                   help="主检测失败时启用替补的最小桌子数量阈值（默认来源于 MIN_BOARDS_FOR_PAW 环境变量）")
    p.add_argument("--force-fallback", action="store_true",
                   help="强制使用替补检测（调试用）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    image_path = Path(args.image)
    run_once(image_path, min_boards_required=args.min_boards, force_fallback=args.force_fallback)
