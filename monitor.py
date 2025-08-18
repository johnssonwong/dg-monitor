# monitor.py
import os, sys, json, time, math, subprocess, shutil
from datetime import datetime, timezone, timedelta
import numpy as np
import cv2
from PIL import Image
import requests

# ---------- ========== CONFIG ========== ----------
# YOUR provided Telegram (auto-filled)
DEFAULT_TELEGRAM_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
DEFAULT_CHAT_ID = "485427847"

# Default DG links (per your request)
DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# Image processing thresholds (can be tuned)
BLUE_HSV_LOW = np.array([90, 50, 50])
BLUE_HSV_HIGH = np.array([140, 255, 255])
RED_HSV_LOW1 = np.array([0, 50, 50])
RED_HSV_HIGH1 = np.array([10, 255, 255])
RED_HSV_LOW2 = np.array([160, 50, 50])
RED_HSV_HIGH2 = np.array([179, 255, 255])

# How many consecutive runs of single-jumps to ignore for "放水判定"
IGNORE_SINGLE_JUMP_CONSECUTIVE = 3

# Where to save screenshots / state
OUT_DIR = "work"
os.makedirs(OUT_DIR, exist_ok=True)
STATE_FILE = "state.json"

# Telegram usage (you can leave as defaults or set via env)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or DEFAULT_TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or DEFAULT_CHAT_ID

# -------- Table bounding boxes (calibration required) ----------
# IMPORTANT: 必须校准：下面是示例模板（x,y,w,h 的 list），
# 每个 entry 对应页面上一个小桌子的 bounding box（在完整截图内的像素坐标）。
# 初次部署时，建议把脚本设为 calibration 模式来产出一张全页截图，然后手动测量每张桌的位置并写入这里。
# 你也可以放入 0 个 box，脚本会尝试 "自动分割" 但不保证稳定。
#
# 示例格式:
TABLE_BOXES = [
    # [x, y, w, h],  # 第1桌 (左上坐标 x,y + 宽高)
    # [x2, y2, w2, h2],
]
# ---------- end CONFIG ----------

# ---------- Helper functions ----------
def send_telegram(text, token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=15)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)

def screenshot_and_save(page, fname):
    page.screenshot(path=fname, full_page=True)

def now_str():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %Z")

# ---------- Image analysis functions ----------
def detect_color_circles(img_bgr):
    """
    基本检测：找出红色和蓝色圆点的位置（返回列表）
    img_bgr: OpenCV image (BGR)
    return: dict with 'blue': [(x,y),...], 'red': [(x,y),...]
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    mask_red1 = cv2.inRange(hsv, RED_HSV_LOW1, RED_HSV_HIGH1)
    mask_red2 = cv2.inRange(hsv, RED_HSV_LOW2, RED_HSV_HIGH2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    # optional: morphological
    kernel = np.ones((3,3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    # detect contours centers
    def centers_from_mask(mask):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers=[]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 8: continue
            (x,y),r = cv2.minEnclosingCircle(c)
            centers.append((int(x),int(y), area))
        return centers
    blue = centers_from_mask(mask_blue)
    red = centers_from_mask(mask_red)
    return {'blue': blue, 'red': red, 'mask_sample': None}

def analyze_table_grid(img_table):
    """
    对单桌进行简单的“连”统计：
    - 我们把每一列上从上至下的圆点按 x 聚合，找到列序列后在列上统计连续颜色 run。
    返回统计结果字典：max_run, count_long_runs(>=4), count_dragon(>=8), count_super(>=10), single_jump_runs, blank_ratio
    """
    h,w = img_table.shape[:2]
    det = detect_color_circles(img_table)
    pts = []
    for (x,y,area) in det['blue']:
        pts.append((x,y,'P'))  # P for Player (blue)
    for (x,y,area) in det['red']:
        pts.append((x,y,'B'))  # B for Banker (red)
    if not pts:
        return {'max_run':0,'count_long_runs':0,'count_dragon':0,'count_super':0,'single_jump_runs':0,'blank_ratio':1.0,'raw_pts':[]}
    # cluster by x into columns
    xs = sorted(set([p[0] for p in pts]))
    # cluster close xs
    cols = []
    tol = max(8, w//100)  # tolerance
    xs_sorted = sorted(xs)
    groups = []
    cur=[xs_sorted[0]]
    for v in xs_sorted[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur=[v]
    groups.append(cur)
    col_centers = [int(sum(g)/len(g)) for g in groups]
    # build columns: for each center, collect points near that x
    col_points = []
    for cx in col_centers:
        col = [p for p in pts if abs(p[0]-cx) <= tol]
        # sort by y (top->down) and map to sequence of 'P'/'B'
        col_sorted = sorted(col, key=lambda x:x[1])
        seq = [c for (_,_,c) in col_sorted]
        col_points.append(seq)
    # Now convert columns into a single timeline by reading columns left-to-right,
    # for each column take topmost marker as the next "粒" in big road.
    timeline = []
    for seq in col_points:
        if seq:
            timeline.append(seq[0])  # top-most symbol
    # compute runs:
    max_run=0
    curr = None
    curr_count=0
    single_jump_runs = 0
    count_long_runs = 0
    count_dragon = 0
    count_super = 0
    for s in timeline:
        if curr is None or s != curr:
            # close last
            if curr_count>0:
                if curr_count==1:
                    single_jump_runs += 1
                if curr_count>=4:
                    count_long_runs += 1
                if curr_count>=8:
                    count_dragon += 1
                if curr_count>=10:
                    count_super += 1
            curr = s
            curr_count = 1
        else:
            curr_count += 1
        if curr_count > max_run:
            max_run = curr_count
    # close tail
    if curr_count>0:
        if curr_count==1:
            single_jump_runs += 1
        if curr_count>=4:
            count_long_runs += 1
        if curr_count>=8:
            count_dragon += 1
        if curr_count>=10:
            count_super += 1
    # blank_ratio: approximate by no. of points vs expected grid size
    approx_occupancy = len(timeline) / max(1, (w//20))  # heuristic
    blank_ratio = 1.0 - min(1.0, approx_occupancy)
    return {
        'max_run': max_run,
        'count_long_runs': count_long_runs,
        'count_dragon': count_dragon,
        'count_super': count_super,
        'single_jump_runs': single_jump_runs,
        'blank_ratio': blank_ratio,
        'raw_pts': pts,
        'timeline': timeline
    }

# ---------- classify scene ----------
def classify_scene(tables_stats):
    total = len(tables_stats)
    tables_with_dragon = sum(1 for t in tables_stats if t['count_dragon']>0 or t['count_super']>0)
    tables_with_super = sum(1 for t in tables_stats if t['count_super']>0)
    tables_dense_long = sum(1 for t in tables_stats if t['blank_ratio']<0.4 and t['count_long_runs']>=2)
    # 放水子类A（满盘长连）
    if tables_dense_long >= max(3, total//4):  # 至少 3 或許多
        return "放水(强提醒)"
    # 放水子类B（超长龙触发）
    if tables_with_super >=1 and (tables_with_dragon >=2) and (tables_with_super + tables_with_dragon)>=3:
        return "放水(强提醒)"
    # 中等胜率（中上）
    if tables_with_dragon >=2:
        # 再确认是否有多连/连珠 via count_long_runs sum
        if sum(t['count_long_runs'] for t in tables_stats) >= 3:
            return "中等胜率(小提醒)"
    # 假信号过滤
    if tables_with_dragon < 2:
        # platform may be lowering winrate
        return "不提醒(假信号/收割)"
    # default
    return "不提醒(默认)"

# ---------- persistence helpers ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,"r",encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def git_commit_and_push(commit_message="update state"):
    """
    用 GITHUB_TOKEN 的凭证把 state.json 提交回 repo（Actions runner 有 GITHUB_TOKEN）
    workflow 必须 checkout 并保留凭证
    """
    try:
        subprocess.run(["git","config","user.email","actions@github.com"], check=True)
        subprocess.run(["git","config","user.name","github-actions"], check=True)
        subprocess.run(["git","add",STATE_FILE], check=True)
        subprocess.run(["git","commit","-m", commit_message], check=True)
        # push uses existing origin with token (checkout persisted credentials)
        subprocess.run(["git","push","origin","HEAD"], check=True)
    except Exception as e:
        print("git push failed:", e)

# ---------- main routine ----------
def main():
    from playwright.sync_api import sync_playwright
    state = load_state()
    # ensure state keys
    if 'alert' not in state:
        state['alert'] = None  # one of None / "放水(强提醒)" / "中等胜率(小提醒)"
    if 'alert_start' not in state:
        state['alert_start'] = None
    # start browser and navigate
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width":1280,"height":800})
        page = context.new_page()
        # try both links until one works
        opened = False
        for url in DG_LINKS:
            try:
                page.goto(url, timeout=45000)
                opened = True
                break
            except Exception as e:
                print("open fail", url, e)
        if not opened:
            print("Cannot open DG links.")
            return
        time.sleep(2)
        # try click "Free" or "免费试玩"
        try:
            selectors = ["text=Free", "text=免费试玩", "text=免费", "button:has-text(\"Free\")"]
            clicked=False
            for sel in selectors:
                try:
                    el = page.query_selector(sel)
                    if el:
                        el.click(timeout=5000)
                        clicked=True
                        break
                except:
                    pass
            # if not clickable, try clicking at common coordinates
            if not clicked:
                # attempt to click roughly middle area to trigger free demo popup
                page.mouse.click(1200,300)
            time.sleep(3)
            # now try to find slider; simulate drag if found
            # common slider class detection attempts (this must be tuned if site changes)
            slider_selectors = ["#slider", ".slider", ".verify-slider", "div[aria-label*='slider']"]
            sl_found=False
            for s in slider_selectors:
                try:
                    ss = page.query_selector(s)
                    if ss:
                        box = ss.bounding_box()
                        if box:
                            x = box['x'] + 5
                            y = box['y'] + box['height']/2
                            page.mouse.move(x,y)
                            page.mouse.down()
                            page.mouse.move(x+box['width']*0.9, y, steps=30)
                            page.mouse.up()
                            sl_found=True
                            time.sleep(2)
                            break
                except:
                    continue
            # fallback: try dragging by fixed coordinate heuristic
            if not sl_found:
                try:
                    # try a common slider location
                    page.mouse.move(400, 500)
                    page.mouse.down()
                    page.mouse.move(1000, 500, steps=30)
                    page.mouse.up()
                    time.sleep(2)
                except:
                    pass
        except Exception as e:
            print("click free/slider fail:", e)
        # wait a bit for lobby to load
        time.sleep(6)
        # take screenshot
        shot = os.path.join(OUT_DIR, f"snap_{int(time.time())}.png")
        try:
            page.screenshot(path=shot, full_page=True)
        except Exception as e:
            print("screenshot fail:", e)
            page.screenshot(path=shot, full_page=False)
        # analyze screenshot
        img = cv2.imread(shot)
        h,w = img.shape[:2]
        print("screenshot size", w, h)
        table_stats = []
        if TABLE_BOXES and len(TABLE_BOXES)>0:
            # use calibrated boxes
            for i,box in enumerate(TABLE_BOXES):
                x,y,ww,hh = box
                x2 = min(w, x+ww)
                y2 = min(h, y+hh)
                crop = img[y:y2, x:x2]
                stats = analyze_table_grid(crop)
                stats['box'] = box
                table_stats.append(stats)
        else:
            # attempt auto-split: try a default grid split (4 columns x 4 rows)
            cols = 4
            rows = max(1, (h//200))  # heuristic
            c_w = w//cols
            r_h = h//rows
            for ry in range(rows):
                for cx in range(cols):
                    x = cx*c_w
                    y = ry*r_h
                    crop = img[y:y+r_h, x:x+c_w]
                    stats = analyze_table_grid(crop)
                    stats['box'] = [x,y,c_w,r_h]
                    table_stats.append(stats)
        # classify
        scene = classify_scene(table_stats)
        print("scene:", scene)
        # alert/persistence logic:
        prev_alert = state.get('alert')
        alert_start = state.get('alert_start')
        now_ts = int(time.time())
        if scene in ["放水(强提醒)","中等胜率(小提醒)"]:
            if prev_alert != scene:
                # new alert started
                state['alert'] = scene
                state['alert_start'] = now_ts
                save_state(state)
                try:
                    git_commit_and_push(f"alert start {scene} at {now_str()}")
                except:
                    pass
                # send telegram message (start)
                msg = f"🔥 <b>{scene}</b> 被偵測到！\n開始時間： {now_str()}\n估計結束時間：等待系統偵測中（會於結束時通知）\n說明：符合你的放水/中等勝率判定。\n來源：{DG_LINKS[0]}"
                send_telegram(msg)
            else:
                # already in alert; do nothing
                print("alert already active:", prev_alert)
        else:
            # scene is not alert
            if prev_alert is not None:
                # previously alert ended — compute duration and notify
                start = int(alert_start) if alert_start else now_ts
                duration_min = (now_ts - start)/60.0
                duration_min = round(duration_min,1)
                msg = f"✅ <b>放水已結束</b>\n類型：{prev_alert}\n開始：{datetime.fromtimestamp(start, timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n結束：{now_str()}\n共持續：{duration_min} 分鐘"
                send_telegram(msg)
                # clear state
                state['alert'] = None
                state['alert_start'] = None
                save_state(state)
                try:
                    git_commit_and_push(f"alert end {prev_alert} at {now_str()} dur {duration_min}m")
                except:
                    pass
            else:
                print("no active alert and nothing to do")
        # cleanup
        browser.close()

if __name__ == "__main__":
    main()
