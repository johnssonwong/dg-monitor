#!/usr/bin/env python3
# main.py — 强化版（优先替补检测 + 历史判定直接触发 Telegram）
# 保留文件名：state.json, history_db.json, history_stats.json, last_summary.json
# 环境变量：TG_BOT_TOKEN, TG_CHAT_ID, MIN_BOARDS_FOR_PAW, HISTORY_LOOKBACK_DAYS, HISTORY_PROB_THRESHOLD

import os, sys, json, logging, argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import cv2, numpy as np, requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

# ------------- 配置 & 常量 -------------
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
MIN_BOARDS_FOR_PAW = int(os.getenv("MIN_BOARDS_FOR_PAW", "3"))
HISTORY_LOOKBACK_DAYS = int(os.getenv("HISTORY_LOOKBACK_DAYS", "28"))
HISTORY_PROB_THRESHOLD = float(os.getenv("HISTORY_PROB_THRESHOLD", "0.35"))
HISTORY_WINDOW_RUNS = int(os.getenv("HISTORY_WINDOW_RUNS", "50"))  # 最近 runs 个数用于历史判定
DEFAULT_LOCAL_IMAGE = Path("/mnt/data/3D04749B-1CDC-42B0-8468-E233F3F81987.jpeg")

STATE_PATH = Path("state.json")
HISTORY_DB_PATH = Path("history_db.json")
HISTORY_STATS_PATH = Path("history_stats.json")
LAST_SUMMARY_PATH = Path("last_summary.json")

# ------------- Logging -------------
logger = logging.getLogger("dg-monitor")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)

BoardRect = Tuple[int, int, int, int, float]

# ------------- JSON helpers -------------
def load_json_safe(p: Path, default):
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"读取 {p} 失败: {e}")
    return default

def save_json_safe(p: Path, data):
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入 {p} 失败: {e}")

def ensure_placeholders():
    if not STATE_PATH.exists():
        save_json_safe(STATE_PATH, {"created": True, "ts": datetime.now().isoformat()})
    if not HISTORY_DB_PATH.exists():
        save_json_safe(HISTORY_DB_PATH, {"runs": []})
    if not HISTORY_STATS_PATH.exists():
        save_json_safe(HISTORY_STATS_PATH, {"total_runs": 0, "total_boards": 0, "pumping_count": 0})
    if not LAST_SUMMARY_PATH.exists():
        save_json_safe(LAST_SUMMARY_PATH, {"ts": datetime.now().isoformat(), "summary": ""})

# ------------- Image detection helpers -------------
def primary_detect_grid(img: np.ndarray, min_area: int = 20000) -> List[BoardRect]:
    """主检测：寻找大矩形（保守）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if w < 80 or h < 50: continue
        rects.append((x,y,w,h,float(area)))
    rects.sort(key=lambda r:(r[1], r[0]))
    return rects

def fallback_detect_edges(img: np.ndarray, min_area: int = 2000) -> List[BoardRect]:
    """替补检测：更激进，优先使用"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        if w < 40 or h < 40: continue
        rects.append((x,y,w,h,float(area)))
    rects.sort(key=lambda r:(r[1], r[0]))
    return rects

def aggressive_multi_pass_detect(img: np.ndarray) -> Tuple[List[BoardRect], str]:
    """多参数多通道尝试：优先使用替补检测（你要求的替补优先），若替补未达到期望再尝试主检测和调整参数。"""
    # 1) 替补（首选）
    fb = fallback_detect_edges(img)
    if len(fb) >= MIN_BOARDS_FOR_PAW:
        return fb, "fallback_primary"
    # 2) 主检测（保守）
    prim = primary_detect_grid(img)
    # 3) 替补+主的合并（避免漏判）
    combined = sorted(fb + prim, key=lambda r:(r[1], r[0]))
    # 去重（合并重叠）
    final = []
    for r in combined:
        x,y,w,h,a = r
        overlap = False
        for fr in final:
            fx,fy,fw,fh,fa = fr
            if (x < fx+fw and fx < x+w and y < fy+fh and fy < y+h):
                overlap = True
                break
        if not overlap:
            final.append(r)
    if len(final) >= MIN_BOARDS_FOR_PAW:
        return final, "fallback_then_primary_combined"
    # 4) 如果都不够，尝试放宽主检测参数（更激进）
    prim2 = primary_detect_grid(img, min_area=8000)
    if len(prim2) >= MIN_BOARDS_FOR_PAW:
        return prim2, "primary_relaxed"
    # 5) 最后返回目前检测到的（可能为 0）
    if len(fb) >= len(prim):
        return fb, "fallback_best"
    return prim, "primary_best"

# ------------- Color ratio & pumping detection -------------
def board_red_blue_ratio(crop: np.ndarray):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 红色范围（两段）
    lower_r1 = np.array([0, 70, 30]); upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([160,70,30]); upper_r2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
    # 蓝色范围
    lower_b = np.array([90,50,30]); upper_b = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)
    # 平滑并计数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)
    rc = int(cv2.countNonZero(mask_r))
    bc = int(cv2.countNonZero(mask_b))
    return rc, bc

def detect_pumping_from_rects(img: np.ndarray, rects: List[BoardRect], prob_threshold: float = HISTORY_PROB_THRESHOLD):
    per = []
    tot_r = 0; tot_b = 0; valid_boards = 0
    for i, (x,y,w,h,a) in enumerate(rects):
        crop = img[y:y+h, x:x+w]
        r,b = board_red_blue_ratio(crop)
        if r + b < 40:
            per.append({"idx": i+1, "red": r, "blue": b, "ratio": None, "valid": False})
            continue
        ratio = r / (r + b)
        per.append({"idx": i+1, "red": r, "blue": b, "ratio": ratio, "valid": True})
        tot_r += r; tot_b += b; valid_boards += 1
    overall_ratio = None
    if tot_r + tot_b > 0:
        overall_ratio = tot_r / (tot_r + tot_b)
    is_pumping = False; reasons = []
    if overall_ratio is not None:
        if overall_ratio >= 0.5 + prob_threshold:
            is_pumping = True; reasons.append(f"整体偏红 (庄) {overall_ratio:.2f} >= 0.5+{prob_threshold}")
        if overall_ratio <= 0.5 - prob_threshold:
            is_pumping = True; reasons.append(f"整体偏蓝 (闲) {overall_ratio:.2f} <= 0.5-{prob_threshold}")
    valid_ratios = [p["ratio"] for p in per if p["valid"] and p["ratio"] is not None]
    if valid_ratios:
        red_biased = sum(1 for r in valid_ratios if r >= 0.5 + prob_threshold)
        blue_biased = sum(1 for r in valid_ratios if r <= 0.5 - prob_threshold)
        if valid_ratios and red_biased / len(valid_ratios) >= 0.6:
            is_pumping = True; reasons.append(f"{red_biased}/{len(valid_ratios)} 桌偏红 (>=60%)")
        if valid_ratios and blue_biased / len(valid_ratios) >= 0.6:
            is_pumping = True; reasons.append(f"{blue_biased}/{len(valid_ratios)} 桌偏蓝 (>=60%)")
    return is_pumping, {"overall_ratio": overall_ratio, "boards_with_data": valid_boards, "per_board": per, "reasons": reasons}

# ------------- Telegram -------------
def send_telegram(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logger.warning("TG_BOT_TOKEN 或 TG_CHAT_ID 未设置，无法发送通知")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram 已发送")
            return True
        logger.warning(f"Telegram 发送失败 HTTP {r.status_code} - {r.text}")
    except Exception as e:
        logger.warning(f"Telegram 发送异常: {e}")
    return False

# ------------- Playwright capture: 多次尝试 -------------
def capture_with_retries(save_path: Path, tries=3):
    """尝试多次打开不同 URL / viewport / 轻微滚动来尽量抓到盘面截图"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    urls = ["https://dg18.co/wap/", "https://dg18.co/", "https://dg18.co/wap/?r=1"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width":1280, "height":1920}, user_agent=headers["User-Agent"])
        page = context.new_page()
        last_exc = None
        for u in urls:
            for t in range(tries):
                try:
                    logger.info(f"截图尝试 URL={u} 第 {t+1}/{tries}")
                    page.goto(u, timeout=10000)
                    # 尝试点击 Free / 点击可能的弹窗
                    try:
                        page.locator("text=Free").click(timeout=2000)
                    except Exception:
                        pass
                    page.wait_for_timeout(400 + 200*t)
                    # 轻微滚动组合尝试
                    if t % 2 == 1:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight/4)")
                        page.wait_for_timeout(300)
                    page.screenshot(path=str(save_path), full_page=True)
                    logger.info(f"截图成功: {save_path}")
                    browser.close()
                    return True, u
                except Exception as e:
                    last_exc = e
                    logger.warning(f"截图失败: {e}")
                    continue
        browser.close()
        logger.error(f"所有截图尝试失败: {last_exc}")
        return False, None

# ------------- 历史判定 -------------
def historical_pumping_check(history_db: dict, lookback_runs: int = HISTORY_WINDOW_RUNS, threshold: float = HISTORY_PROB_THRESHOLD):
    """计算最近 lookback_runs 次是否有足够比例判定为 pumping"""
    runs = history_db.get("runs", [])
    if not runs:
        return False, {"reason":"无历史数据"}
    # 取最近 N runs
    recent = runs[-lookback_runs:]
    total = len(recent)
    pumping_cnt = sum(1 for r in recent if r.get("is_pumping"))
    if total == 0:
        return False, {"reason":"无历史数据2"}
    frac = pumping_cnt / total
    reason = f"最近 {total} 次中 {pumping_cnt} 次被判為放水 (比例 {frac:.2f})"
    return (frac >= threshold), {"fraction": frac, "pumping_cnt": pumping_cnt, "total": total, "reason": reason}

# ------------- 主流程 -------------
def run_once(image_path: Path, no_fetch: bool = False, force_fallback: bool = False):
    logger.info("=== DG monitor run start ===")
    ensure_placeholders()

    screenshot_path = image_path
    if not no_fetch:
        ok, used_url = capture_with_retries(screenshot_path, tries=3)
        if not ok:
            logger.warning("抓取站点失败，尝试使用本地图片作为回退")
            if not screenshot_path.exists():
                logger.error("无可用图片，退出")
                return
    else:
        if not screenshot_path.exists():
            logger.error("指定本地图片不存在，退出")
            return

    # 读取图片（对 Windows 路径和非 UTF 路径更稳）
    img = cv2.imdecode(np.fromfile(str(screenshot_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.error("无法读取图片文件")
        return

    # 如果强制替补或优先替补（按你的要求：替补优先）
    if force_fallback:
        rects, method = aggressive_multi_pass_detect(img)
    else:
        # 优先尝试替补（fallback），如不足再合并/主检测等
        rects, method = aggressive_multi_pass_detect(img)

    logger.info(f"检测方法={method}, 检到桌子数={len(rects)} (阈值 {MIN_BOARDS_FOR_PAW})")

    # 分析每桌颜色比例
    is_pumping, details = detect_pumping_from_rects(img, rects, prob_threshold=HISTORY_PROB_THRESHOLD)

    # 历史判定（从 history_db.json）
    history_db = load_json_safe(HISTORY_DB_PATH, {"runs": []})
    hist_flag, hist_details = historical_pumping_check(history_db, lookback_runs=HISTORY_WINDOW_RUNS, threshold=HISTORY_PROB_THRESHOLD)

    # 准备要写入历史的一条记录
    now_ts = datetime.now().isoformat()
    run_entry = {
        "ts": now_ts,
        "method": method,
        "detected_boards": len(rects),
        "is_pumping": bool(is_pumping),
        "overall_ratio": details.get("overall_ratio")
    }
    # 追加到历史并保存
    history_db.setdefault("runs", []).append(run_entry)
    # 保持历史长度合理
    if len(history_db["runs"]) > 2000:
        history_db["runs"] = history_db["runs"][-2000:]
    save_json_safe(HISTORY_DB_PATH, history_db)

    # 更新 stats
    stats = load_json_safe(HISTORY_STATS_PATH, {"total_runs":0, "total_boards":0, "pumping_count":0})
    stats["total_runs"] = stats.get("total_runs", 0) + 1
    stats["total_boards"] = stats.get("total_boards", 0) + len(rects)
    stats["pumping_count"] = stats.get("pumping_count", 0) + (1 if is_pumping else 0)
    save_json_safe(HISTORY_STATS_PATH, stats)

    # 保存 state/last_summary
    state = load_json_safe(STATE_PATH, {})
    state.update({
        "last_run": now_ts,
        "last_method": method,
        "last_detected_boards": len(rects),
        "last_is_pumping": bool(is_pumping)
    })
    save_json_safe(STATE_PATH, state)

    last_summary = {"ts": now_ts, "summary": f"桌数={len(rects)} method={method} pumping={bool(is_pumping)}"}
    save_json_safe(LAST_SUMMARY_PATH, last_summary)

    # 决策：只要「实时判定为放水」或「历史判定阈值触发」就立即发 Telegram
    should_notify = False
    notify_reasons = []
    if is_pumping:
        should_notify = True
        notify_reasons.append("实时判定：本次检测发现放水倾向")
    if hist_flag:
        should_notify = True
        notify_reasons.append("历史判定：最近历史数据比例达到阈值")

    if should_notify:
        # 构造消息（保持你要求格式）
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall = details.get("overall_ratio")
        boards_with_data = details.get("boards_with_data")
        lines = [
            f"<b>DG 放水警报</b> — {ts}",
            f"检测方法: {method}",
            f"检测到桌子数: {len(rects)}; 有效桌数: {boards_with_data}",
            f"整体庄(红)占比: {overall:.2f}" if overall is not None else "整体占比: 无数据",
            "判定原因: " + " ; ".join(notify_reasons),
            "",
            "每桌详情（显示 ratio 或 数据不足）:"
        ]
        for b in details["per_board"]:
            if b["valid"] and b["ratio"] is not None:
                lines.append(f"桌 {b['idx']}: 庄占比 {b['ratio']:.2f}")
            else:
                lines.append(f"桌 {b['idx']}: 数据不足")
        if hist_flag:
            lines.append("")
            lines.append(f"历史判定细节: {hist_details.get('reason')}")
        message = "\n".join(lines)
        sent = send_telegram(message)
        if not sent:
            logger.warning("通知发送失败，请检查 TG_BOT_TOKEN / TG_CHAT_ID")
    else:
        logger.info("静默：既未实时判定放水，亦未满足历史判定阈值（不发通知）")

    logger.info("=== DG monitor run end ===")
    # 确保退出码 0 避免 CI 误判
    try:
        sys.exit(0)
    except SystemExit:
        pass

# ------------- CLI -------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", default=str(DEFAULT_LOCAL_IMAGE), help="本地图片路径（若不想从网站抓取传此参数）")
    p.add_argument("--no-fetch", action="store_true", help="不抓取网站，直接使用本地图片")
    p.add_argument("--force-fallback", action="store_true", help="强制替补检测（优先替补）")
    return p.parse_args()

if __name__ == "__main__":
    ensure_placeholders()
    args = parse_args()
    try:
        run_once(Path(args.image), no_fetch=args.no_fetch, force_fallback=args.force_fallback)
    except Exception as e:
        logger.exception(f"未捕获异常: {e}")
        st = load_json_safe(STATE_PATH, {})
        st.update({"error": str(e), "error_ts": datetime.now().isoformat()})
        save_json_safe(STATE_PATH, st)
        ensure_placeholders()
        # 保持 exit 0，避免 CI 失败（如你之前要求）
        try:
            sys.exit(0)
        except SystemExit:
            pass
