# detect_dg.py
# Run on GitHub Actions every 5 minutes
import os
import sys
import time
import json
import math
import base64
import subprocess
from datetime import datetime, timezone, timedelta

# Playwright + OpenCV
from playwright.sync_api import sync_playwright
import cv2
import numpy as np
import requests

# ---------------------------
# Config: these read from env (set in GitHub Secrets)
# ---------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")  # e.g. 8134... (put into repo secret)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # e.g. 485427847
DG_URL_1 = os.environ.get("DG_URL_1", "https://dg18.co/wap/")
DG_URL_2 = os.environ.get("DG_URL_2", "https://dg18.co/")
# commit/persist config
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Actions builtin token for committing state

# Parameters / thresholds (可根据样本调)
MIN_LONG = 4        # 连续>=4 = 长连
MIN_DRAGON = 8      # 连续>=8 = 长龙
MIN_SUPER = 10      # 连续>=10 = 超长龙
MIN_TABLES_FOR_PUSHD = 3  # 放水必须至少 3 张桌子符合（子类要求）
SINGLE_JUMP_IGNORE = 3  # 连续3次单跳不计入放水判定（采样逻辑里会忽略短时序噪声）

# state file in repo
STATE_FILE = "dg_monitor_state.json"

# ---------------------------
# Utility: Telegram
# ---------------------------
def send_telegram(text, image_path=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat id not set. Skipping telegram send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        # send text first (to ensure message arrives even if image fails)
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=20)
        r.raise_for_status()
        if image_path and os.path.exists(image_path):
            # send image
            files = {"photo": open(image_path, "rb")}
            url2 = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            payload2 = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
            r2 = requests.post(url2, data=payload2, files=files, timeout=30)
            try:
                r2.raise_for_status()
            except:
                print("Send photo failed:", r2.text)
    except Exception as e:
        print("Telegram send failed:", e)

# ---------------------------
# State persistence (commit state file back to repo)
# ---------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf8") as f:
            return json.load(f)
    else:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    # commit the state file back so subsequent runs can read it
    # Use git with GITHUB_TOKEN to push commit (Actions provides GITHUB_TOKEN)
    try:
        repo = os.environ.get("GITHUB_REPOSITORY")  # e.g. user/repo
        if repo and GITHUB_TOKEN:
            subprocess.run(["git", "config", "user.email", "action@github.com"], check=True)
            subprocess.run(["git", "config", "user.name", "github-action"], check=True)
            subprocess.run(["git", "add", STATE_FILE], check=True)
            subprocess.run(["git", "commit", "-m", "update monitor state"], check=True)
            remote = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{repo}.git"
            subprocess.run(["git", "push", remote, "HEAD:main"], check=True, timeout=60)
    except Exception as e:
        print("Warning: could not commit state to repo:", e)

# ---------------------------
# Image processing helpers
# ---------------------------
def detect_circles_and_colors(img):
    """
    Input: BGR image of a table area (cropped)
    Output: list of detected marks with (x,y,color) where color in {"B","R","T","G"}:
      B = Player (blue), R = Banker (red), T = Tie/green marker maybe.
    Method: color thresholding to find blue/red circles.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # thresholds (may need tuning)
    # blue
    lower_blue = np.array([90, 80, 60])
    upper_blue = np.array([135, 255, 255])
    # red - handle both ranges
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 50])
    upper_red2 = np.array([179, 255, 255])
    # green (tie marker)
    lower_green = np.array([40, 60, 50])
    upper_green = np.array([90, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # use simple blob detection on each mask
    kernel = np.ones((3,3), np.uint8)
    masks = [("B", mask_blue), ("R", mask_red), ("T", mask_green)]
    marks = []
    for color_label, mask in masks:
        # clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # noise
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            marks.append({"x": int(x), "y": int(y), "c": color_label, "area": area})
    return marks

def cluster_grid_positions(marks, max_cols=20, max_rows=20):
    """
    From detected marks, cluster by approximate X positions and Y positions to find grid columns/rows.
    Returns grid mapping: grid[row][col] = color or None
    """
    if not marks:
        return [], []
    xs = np.array([m["x"] for m in marks]).reshape(-1,1)
    ys = np.array([m["y"] for m in marks]).reshape(-1,1)

    # cluster x into columns using kmeans heuristic: try to find column centers by 1D clustering
    def cluster_1d(vals, max_k=20):
        vals = np.sort(vals.flatten())
        if len(vals) == 0:
            return []
        # try to find gaps, simple approach: use fixed binning by quantiles
        # compute unique positions by rounding
        uniq = np.unique(np.round(vals, -0))  # round to integer
        # attempt to cluster by distance: find big gaps
        diffs = np.diff(uniq)
        # if many gaps > threshold then split
        # but simpler: use hierarchical clustering via distance threshold
        clusters = []
        current = [uniq[0]]
        for d, val in zip(diffs, uniq[1:]):
            if d > 20:  # gap threshold, may need tuning
                clusters.append(np.mean(current))
                current = [val]
            else:
                current.append(val)
        if current:
            clusters.append(np.mean(current))
        return clusters

    x_centers = cluster_1d(xs)
    y_centers = cluster_1d(ys)
    # build grid indices by mapping each mark to nearest center
    grid = {}
    for m in marks:
        col = min(range(len(x_centers)), key=lambda i: abs(m["x"]-x_centers[i])) if x_centers else 0
        row = min(range(len(y_centers)), key=lambda i: abs(m["y"]-y_centers[i])) if y_centers else 0
        grid.setdefault(row, {})
        # choose priority if multiple marks in same cell (take larger area)
        prev = grid[row].get(col)
        if prev is None or m["area"] > prev["area"]:
            grid[row][col] = m
    # produce matrix
    max_r = max(grid.keys()) if grid else -1
    max_c = max((max(row.keys()) if row else -1) for row in grid.values()) if grid else -1
    mat = []
    for r in range(max_r+1):
        row = []
        for c in range(max_c+1):
            cell = grid.get(r, {}).get(c)
            row.append(cell["c"] if cell else None)
        mat.append(row)
    return mat, {"rows": len(y_centers), "cols": len(x_centers)}

# ---------------------------
# Table classification (per user rules)
# ---------------------------
def analyze_table_grid(grid_matrix):
    """
    Input: grid_matrix: list of rows, each row is list of 'B'/'R'/'T'/None
    Returns stats: max_run, count_long_runs (>=4), count_dragon (>=8), count_super, single_jump_count, blank_ratio, 连珠_count
    Note: counting rules: must be "same-side连续在同一排"...
    We'll scan each row top-to-bottom, left-to-right per your instruction and compute runs along columns in the standard road order:
    """
    rows = len(grid_matrix)
    cols = max((len(r) for r in grid_matrix), default=0)
    total_cells = rows*cols if rows>0 and cols>0 else 1
    blank_count = 0
    runs = []
    # For each column (left->right), scan top->bottom to follow the '大路' reading order
    for c in range(cols):
        prev = None
        runlen = 0
        for r in range(rows):
            val = grid_matrix[r][c] if c < len(grid_matrix[r]) else None
            if val is None:
                blank_count += 1
            if val == prev and val is not None:
                runlen += 1
            else:
                if prev is not None:
                    runs.append((prev, runlen))
                runlen = 1 if val is not None else 0
                prev = val
        if prev is not None:
            runs.append((prev, runlen))
    # compute stats
    max_run = max((length for (_, length) in runs), default=0)
    count_long_runs = sum(1 for (_, length) in runs if length >= MIN_LONG)
    count_dragon = sum(1 for (_, length) in runs if length >= MIN_DRAGON)
    count_super = sum(1 for (_, length) in runs if length >= MIN_SUPER)
    single_jump = sum(1 for (_, length) in runs if length == 1)
    blank_ratio = blank_count / total_cells
    # 简单连珠检测：查找同一 run 连续在下一排也 >=4 （这里我们近似检测：若连在同列连续出现4并且在相邻列同位置也有>=4）
    连珠_count = 0
    # naive 连珠检测 (可改进)
    # return
    return {
        "max_run": max_run,
        "count_long_runs": count_long_runs,
        "count_dragon": count_dragon,
        "count_super": count_super,
        "single_jump": single_jump,
        "blank_ratio": blank_ratio,
        "连珠_count": 连珠_count,
        "rows": rows,
        "cols": cols,
        "runs": runs
    }

# ---------------------------
# Scene classification per your saved rules
# ---------------------------
def classify_scene(all_table_stats):
    total = len(all_table_stats)
    tables_with_dragon = sum(1 for t in all_table_stats if t["count_dragon"]>0 or t["count_super"]>0)
    tables_with_super = sum(1 for t in all_table_stats if t["count_super"]>0)
    tables_dense_long = sum(1 for t in all_table_stats if t["blank_ratio"]<0.4 and t["count_long_runs"]>=2)
    # 放水判定A
    if tables_dense_long >= MIN_TABLES_FOR_PUSHD:
        return "放水(强提醒)", {"type":"满盘长连局势型","tables_dense_long":tables_dense_long}
    # 放水判定B 超长龙触发型
    if tables_with_super >=1 and tables_with_dragon >=2 and (tables_with_super + tables_with_dragon) >= MIN_TABLES_FOR_PUSHD:
        return "放水(强提醒)", {"type":"超长龙触发型","tables_with_super":tables_with_super,"tables_with_dragon":tables_with_dragon}
    # 中等胜率（中上）
    if tables_with_dragon >= 2 and any(t["连珠_count"]>0 for t in all_table_stats):
        return "中等胜率(小提醒)", {"type":"中等(多连/连珠+>=2长龙)","tables_with_dragon":tables_with_dragon}
    # 假信号过滤
    if tables_with_dragon < 2:
        # 判断为收割/不提醒
        return "不提醒(假信号/收割)", {"tables_with_dragon":tables_with_dragon}
    # 其他收割/胜率中等
    # heuristics: many blanks or many single jumps
    blanks_high = sum(1 for t in all_table_stats if t["blank_ratio"]>0.6)
    singles_high = sum(1 for t in all_table_stats if t["single_jump"] > 6)
    if blanks_high > total/2 or singles_high > total/2:
        return "不提醒(收割或胜率中等)", {"blanks_high":blanks_high,"singles_high":singles_high}
    return "不提醒(默认)", {}

# ---------------------------
# Main: Playwright flow to access DG, click Free, slide, screenshot
# ---------------------------
def run_cycle():
    state = load_state()
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))  # Malaysia UTC+8
    now_str = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    print("Run at", now_str)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1366,"height":768})
        # try first link
        try:
            page.goto(DG_URL_1, timeout=45000)
        except Exception as e:
            print("Primary URL failed, trying fallback:", e)
            page.goto(DG_URL_2, timeout=45000)
        time.sleep(2)
        # Try to click "Free" or "免费试玩" button; selector may vary
        clicked = False
        for sel in ["text=Free", "text=免费试玩", "button:has-text(\"Free\")", "a:has-text(\"Free\")", "text=试玩"]:
            try:
                el = page.query_selector(sel)
                if el:
                    el.click(timeout=5000)
                    clicked = True
                    time.sleep(1)
                    break
            except:
                pass
        # If slider appears (simple heuristic: wait for an element that looks like slider)
        time.sleep(2)
        # attempt to find slider element
        try:
            slider = page.query_selector("div[class*='slider'], #slider, .drag, .verify") or page.query_selector("div:has(:text('slide'))")
            if slider:
                box = slider.bounding_box()
                if box:
                    sx = box["x"] + 5
                    sy = box["y"] + box["height"]/2
                    page.mouse.move(sx, sy)
                    page.mouse.down()
                    page.mouse.move(sx + box["width"]-10, sy, steps=20)
                    page.mouse.up()
                    time.sleep(1)
        except Exception as e:
            print("Slider handling attempt fail:", e)

        # wait for some game elements (timeout)
        time.sleep(3)
        # take full screenshot as debug
        page.screenshot(path="debug_full.png", full_page=True)
        # Attempt to find table elements; common patterns: canvas, img, divs with baccarat content.
        tables = page.query_selector_all("canvas, img, .baccarat-board, .game-table, .table-item")
        crops = []
        if tables:
            for i, el in enumerate(tables):
                bb = el.bounding_box()
                if bb and bb["width"]>50 and bb["height"]>40:
                    x,y,w,h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
                    crops.append((x,y,w,h))
        else:
            # fallback: use manual grid layout heuristics — try to crop by screen regions (assume grid 4x3)
            w,h = 1366,768
            cols = 4
            rows = 3
            cw = w//cols
            ch = h//rows
            for r in range(rows):
                for c in range(cols):
                    crops.append((c*cw, r*ch, cw, ch))

        # load debug_full to crop
        img_full = cv2.imread("debug_full.png")
        all_stats = []
        for idx, (x,y,wc,hc) in enumerate(crops):
            crop = img_full[y:y+hc, x:x+wc].copy()
            # optional save for debugging
            cv2.imwrite(f"crop_{idx}.png", crop)
            marks = detect_circles_and_colors(crop)
            grid, dims = cluster_grid_positions(marks)
            stats = analyze_table_grid(grid)
            stats["table_index"] = idx
            stats["marks_count"] = len(marks)
            all_stats.append(stats)

        scene, meta = classify_scene(all_stats)
        print("Classified scene:", scene, meta)

        # Alert logic: follow your rule: only send for 放水(强提醒) or 中等胜率(小提醒)
        last = state.get("last_scene")
        last_changed = (last != scene)
        now_ts = int(time.time())
        # manage start/end times
        if scene.startswith("放水"):
            if state.get("active") != True:
                # start new event
                state["active"] = True
                state["start_time"] = now_ts
                state["scene"] = scene
                send_telegram(f"📣 <b>DG 检测到：{scene}</b>\n时间：{now_str}\n说明：{meta}\n(会附带截图和简要统计)", image_path="debug_full.png")
            else:
                # already active - update (no repeated message)
                pass
        elif scene.startswith("中等胜率"):
            if state.get("active") != True:
                state["active"] = True
                state["start_time"] = now_ts
                state["scene"] = scene
                send_telegram(f"🔔 <b>DG 检测到：{scene}</b>\n时间：{now_str}\n说明：{meta}\n(小提醒)", image_path="debug_full.png")
            else:
                # already active continue
                pass
        else:
            # if previously active, then end and send end message
            if state.get("active"):
                start = state.get("start_time", now_ts)
                duration_minutes = (now_ts - start)//60
                send_telegram(f"✅ <b>放水/提醒结束</b>\n原状态：{state.get('scene')}\n开始时间：{datetime.fromtimestamp(start, tz=timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n结束时间：{now_str}\n共持续：{duration_minutes} 分钟")
            state["active"] = False
            state["scene"] = scene

        state["last_scene"] = scene
        state["last_checked"] = now_str
        # for history, append entry
        history = state.get("history", [])
        history.append({"ts": now_ts, "scene": scene})
        state["history"] = history[-200:]  # keep last 200
        save_state(state)
        browser.close()

if __name__ == "__main__":
    run_cycle()
