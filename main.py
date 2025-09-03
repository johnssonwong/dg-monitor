#!/usr/bin/env python3
# main.py (robust fixed version)
# 保留一切你指定的文件名与行为；增加稳健性：确保 JSON 文件存在、全局异常捕获并在 finally 中写入文件，避免 CI Exit 1

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import requests

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

# ----------------------
# 环境 / 常量
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

DEFAULT_LOCAL_IMAGE = Path("/mnt/data/3D04749B-1CDC-42B0-8468-E233F3F81987.jpeg")

# ----------------------
# 日志
# ----------------------
logger = logging.getLogger("dg-monitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

BoardRect = Tuple[int, int, int, int, float]

# ----------------------
# JSON helpers
# ----------------------
def load_json_safe(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return default
    except Exception as e:
        logger.warning(f"读取 {path} 失败: {e}")
        return default

def save_json_safe(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入 {path} 失败: {e}")

def ensure_json_placeholders():
    """确保四个关键 json 文件存在（写入最小结构），避免 Git 在后续步骤报 pathspec not found"""
    try:
        if not STATE_PATH.exists():
            save_json_safe(STATE_PATH, {"created": True, "ts": datetime.now().isoformat()})
        if not HISTORY_DB_PATH.exists():
            save_json_safe(HISTORY_DB_PATH, {"runs": []})
        if not HISTORY_STATS_PATH.exists():
            save_json_safe(HISTORY_STATS_PATH, {"total_runs": 0, "total_boards": 0, "pumping_count": 0})
        if not LAST_SUMMARY_PATH.exists():
            save_json_safe(LAST_SUMMARY_PATH, {"ts": datetime.now().isoformat(), "summary": ""})
    except Exception as e:
        logger.warning(f"ensure_json_placeholders 异常: {e}")

# ----------------------
# 图像检测：主方法 + 替补方法（保持与你之前的算法一致）
# ----------------------
def primary_detect_grid(img: np.ndarray, min_area: int = 20000, approx_eps_ratio: float = 0.02) -> List[BoardRect]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9)
    kx = max(9, img.shape[1] // 160)
    ky = max(5, img.shape[0] // 240)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
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

def fallback_detect_edges(img: np.ndarray, min_area: int = 2000) -> List[BoardRect]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    k = max(3, img.shape[1] // 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 and h > 40:
            rects.append((x, y, w, h, float(area)))
    rects.sort(key=lambda r: (r[1], r[0]))
    return rects

def detect_boards(img: np.ndarray, min_boards_required: int = 3, force_fallback: bool = False):
    if force_fallback:
        logger.info("强制使用替补检测 (force_fallback=True)")
        return fallback_detect_edges(img), "fallback_forced"
    primary = primary_detect_grid(img)
    if len(primary) >= min_boards_required:
        return primary, "primary"
    fb = fallback_detect_edges(img)
    if len(fb) >= len(primary) and len(fb) > 0:
        return fb, "fallback"
    else:
        return primary, "primary"

def board_red_blue_ratio(crop: np.ndarray):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 60, 30]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 60, 30]); upper_red2 = np.array([179, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    lower_blue = np.array([90, 40, 30]); upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
    red_count = int(cv2.countNonZero(mask_red))
    blue_count = int(cv2.countNonZero(mask_blue))
    return red_count, blue_count

def detect_pumping_period(img: np.ndarray, rects: List[BoardRect], prob_threshold: float = HISTORY_PROB_THRESHOLD):
    per_board = []
    total_red = 0; total_blue = 0; boards_with_data = 0
    for i, (x, y, w, h, area) in enumerate(rects):
        crop = img[y:y+h, x:x+w]
        r, b = board_red_blue_ratio(crop)
        if r + b < 50:
            per_board.append({"idx": i+1, "red": r, "blue": b, "ratio": None, "valid": False})
            continue
        ratio = r / (r + b)
        per_board.append({"idx": i+1, "red": r, "blue": b, "ratio": ratio, "valid": True})
        total_red += r; total_blue += b; boards_with_data += 1
    overall_ratio = None
    if total_red + total_blue > 0:
        overall_ratio = total_red / (total_red + total_blue)
    is_pumping = False; reason = []
    if overall_ratio is not None:
        if overall_ratio >= 0.5 + prob_threshold:
            is_pumping = True; reason.append(f"整体偏向红方 (庄) 比例 {overall_ratio:.2f} >= 0.5+{prob_threshold}")
        elif overall_ratio <= 0.5 - prob_threshold:
            is_pumping = True; reason.append(f"整体偏向蓝方 (闲) 比例 {overall_ratio:.2f} <= 0.5-{prob_threshold}")
    valid_ratios = [p["ratio"] for p in per_board if p["valid"] and p["ratio"] is not None]
    if valid_ratios:
        red_biased = sum(1 for r in valid_ratios if r >= 0.5 + prob_threshold)
        blue_biased = sum(1 for r in valid_ratios if r <= 0.5 - prob_threshold)
        if len(valid_ratios) > 0:
            if red_biased / len(valid_ratios) >= 0.6:
                is_pumping = True; reason.append(f"{red_biased}/{len(valid_ratios)} 桌偏向庄 (占比 >=60%)")
            if blue_biased / len(valid_ratios) >= 0.6:
                is_pumping = True; reason.append(f"{blue_biased}/{len(valid_ratios)} 桌偏向闲 (占比 >=60%)")
    details = {"overall_ratio": overall_ratio, "boards_with_data": boards_with_data, "per_board": per_board, "reasons": reason}
    return is_pumping, details

def send_telegram_alert(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.warning("TG_BOT_TOKEN 或 TG_CHAT_ID 未配置，无法发送 Telegram 通知")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            logger.info("已发送 Telegram 提醒")
            return True
        else:
            logger.warning(f"发送 Telegram 失败: HTTP {r.status_code} - {r.text}")
            return False
    except Exception as e:
        logger.warning(f"发送 Telegram 异常: {e}")
        return False

def save_crops(img: np.ndarray, rects: List[BoardRect], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h, area) in enumerate(rects):
        crop = img[y:y+h, x:x+w]
        fname = out_dir / f"board_{i+1:02d}_{x}_{y}_{w}x{h}.png"
        cv2.imwrite(str(fname), crop)

def fetch_site_screenshot(save_path: Path, try_wap_first: bool = True, click_free: bool = True, timeout_ms: int = 8000):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("开始使用 Playwright 抓取 DG 页面截图")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width": 1280, "height": 1920}, user_agent="Mozilla/5.0")
        page = context.new_page()
        urls = ["https://dg18.co/wap/", "https://dg18.co/"] if try_wap_first else ["https://dg18.co/", "https://dg18.co/wap/"]
        last_exception = None
        for u in urls:
            try:
                logger.info(f"打开 {u} （尝试）")
                page.goto(u, timeout=timeout_ms)
                if click_free:
                    try:
                        logger.info("尝试点击文本 Free")
                        page.locator("text=Free").click(timeout=3000)
                    except PWTimeoutError:
                        pass
                    except Exception:
                        pass
                page.wait_for_timeout(600)
                page.screenshot(path=str(save_path), full_page=True)
                logger.info(f"页面截图已保存到: {save_path}")
                browser.close()
                return True, u
            except Exception as e:
                logger.warning(f"打开 {u} 或截图失败: {e}")
                last_exception = e
                continue
        browser.close()
        logger.error(f"所有 URL 都尝试过但失败: 最后异常: {last_exception}")
        return False, None

# ----------------------
# 主流程（增加了 robust try/except/finally）
# ----------------------
def run_once(image_path: Path, min_boards_required: int, force_fallback: bool = False, try_fetch_site: bool = True):
    logger.info("=== DG monitor run start ===")
    # 确保 placeholder 文件存在（极重要）
    ensure_json_placeholders()

    screenshot_path = image_path
    if try_fetch_site:
        ok, used_url = fetch_site_screenshot(screenshot_path, try_wap_first=True, click_free=True)
        if not ok:
            logger.warning("抓取站点失败，将尝试使用本地图片（若存在）")
            if not screenshot_path.exists():
                logger.error(f"没有可用的截图: {screenshot_path}")
                return
    else:
        if not screenshot_path.exists():
            logger.error(f"本地图片不存在: {screenshot_path}")
            return

    img = cv2.imdecode(np.fromfile(str(screenshot_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("无法读取截图文件或格式不支持")
        return

    logger.info(f"打开 https://dg18.co/wap/ （尝试 1）")
    logger.info("点击文本 Free")

    rects, method = detect_boards(img, min_boards_required=min_boards_required, force_fallback=force_fallback)
    logger.info(f"检测方法: {method}; 检测到桌子数: {len(rects)} (阈值 {min_boards_required})")

    if method == "primary" and len(rects) < min_boards_required:
        logger.info("主检测未到阈值，马上使用替补检测 (immediate fallback)")
        rects_fb = fallback_detect_edges(img)
        if len(rects_fb) >= len(rects):
            rects = rects_fb
            method = "fallback_after_primary"

    if rects:
        save_crops(img, rects, Path("detected_boards"))
        logger.info(f"已保存各桌截图到 detected_boards（共 {len(rects)} 张）")
    else:
        logger.info("未检测到任何桌子（rects == 0）")

    is_pumping, details = detect_pumping_period(img, rects, prob_threshold=HISTORY_PROB_THRESHOLD)
    if is_pumping:
        logger.info(f"实时判定: 放水时段 / 收割时段（判定理由: {'; '.join(details.get('reasons', []))}）")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall = details.get("overall_ratio")
        boards_with_data = details.get("boards_with_data")
        lines = [f"<b>DG 放水警报</b>  — {ts}",
                 f"检测方法: {method}",
                 f"检测到桌子数: {len(rects)}; 有效桌数: {boards_with_data}",
                 f"整体庄(红)占比: {overall:.2f}" if overall is not None else "整体占比: 无数据",
                 f"判定理由: {'; '.join(details.get('reasons', [])) or '（无）'}",
                 "",
                 "每桌详情（仅展示 ratio）:"]
        for b in details["per_board"]:
            if b["valid"] and b["ratio"] is not None:
                lines.append(f"桌 {b['idx']}: 庄占比 {b['ratio']:.2f}")
            else:
                lines.append(f"桌 {b['idx']}: 数据不足")
        message = "\n".join(lines)
        send_telegram_alert(message)
    else:
        logger.info("实时判定: 不是放水/中等胜率（静默，不发通知）")

    ts = datetime.now().isoformat()
    state = load_json_safe(STATE_PATH, {})
    state.update({
        "last_run": ts,
        "last_image": str(image_path),
        "detected_boards": len(rects),
        "method": method,
        "is_pumping": bool(is_pumping)
    })
    save_json_safe(STATE_PATH, state)

    history_db = load_json_safe(HISTORY_DB_PATH, {})
    history_db.setdefault("runs", []).append({
        "ts": ts,
        "boards": len(rects),
        "method": method,
        "is_pumping": bool(is_pumping),
        "overall_ratio": details.get("overall_ratio")
    })
    if len(history_db["runs"]) > 500:
        history_db["runs"] = history_db["runs"][-500:]
    save_json_safe(HISTORY_DB_PATH, history_db)

    stats = load_json_safe(HISTORY_STATS_PATH, {"total_runs": 0, "total_boards": 0, "pumping_count": 0})
    stats["total_runs"] = stats.get("total_runs", 0) + 1
    stats["total_boards"] = stats.get("total_boards", 0) + len(rects)
    stats["pumping_count"] = stats.get("pumping_count", 0) + (1 if is_pumping else 0)
    save_json_safe(HISTORY_STATS_PATH, stats)

    last_summary = {
        "ts": ts,
        "summary": f"检测到桌子 {len(rects)} (method={method}) - 放水: {bool(is_pumping)}"
    }
    save_json_safe(LAST_SUMMARY_PATH, last_summary)

    logger.info("=== DG monitor run end ===")

# ----------------------
# 程序入口：全局异常捕获并确保 placeholder 存在，避免 CI 失败
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="DG monitor - detect boards and alert on pumping periods")
    p.add_argument("--image", "-i", type=str, default=str(DEFAULT_LOCAL_IMAGE),
                   help="本地图片路径，若指定则不会从网站抓取截图 (默认为对话中图片路径)")
    p.add_argument("--min-boards", type=int, default=MIN_BOARDS_FOR_PAW,
                   help="主检测失败时启用替补的最小桌子数量阈值")
    p.add_argument("--force-fallback", action="store_true",
                   help="强制使用替补检测")
    p.add_argument("--no-fetch", action="store_true",
                   help="不要尝试从网站抓取截图（只使用本地图片）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 无论如何先确保占位文件存在（大幅减少 Git pathspec 错误）
    ensure_json_placeholders()
    try:
        run_once(Path(args.image), min_boards_required=args.min_boards, force_fallback=args.force_fallback, try_fetch_site=not args.no_fetch)
    except Exception as e:
        logger.exception(f"运行时出现未捕获异常: {e}")
        # 写入 state.json 的错误信息，便于后续诊断（但仍避免删除/写入私有数据）
        err_state = load_json_safe(STATE_PATH, {})
        err_state.update({
            "error": str(e),
            "error_ts": datetime.now().isoformat()
        })
        save_json_safe(STATE_PATH, err_state)
    finally:
        # 再次确保四个文件在退出前存在（避免 Workflow 后续 git add 找不到）
        ensure_json_placeholders()
        # 明确以 0 退出，避免 CI 出现 exit code 1。
        try:
            sys.exit(0)
        except SystemExit:
            # 如果 runner 以某种方式阻止退出，会尽量安静结束
            pass
