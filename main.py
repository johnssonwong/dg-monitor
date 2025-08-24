# -*- coding: utf-8 -*-
"""
DG 实盘监测器（GitHub Actions 用）
- Playwright 登录 DG、点击 Free、滑动安全条、截图（真实页面）
- OpenCV 分析每个桌面画面点阵，判定连数、多连/连珠、长龙/超长龙
- 判定总体时段（放水 / 中等胜率（中上） / 胜率中等 / 收割）
- 在“放水”或“中等胜率（中上）”时发送 Telegram 提醒（开始通知 + 估算结束时间）
- 在事件结束时发送 Telegram 结束通知（含真实持续时间）
- 状态存储于 state.json（会被 workflow commit 回 repo）
注意：请在 GitHub Secrets 中设置 TG_BOT_TOKEN 与 TG_CHAT_ID
"""

import os, sys, time, json, math, traceback
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from playwright.sync_api import sync_playwright

# ---------- config ----------
TZ = timezone(timedelta(hours=8))  # Malaysia UTC+8
STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"

# DG links (you gave)
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]

# env / secrets
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()

# --- 读取环境变量（稳健处理空字符串 / 非法值） ---
def safe_int_env(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    # strip whitespace; if empty -> default
    v_str = str(v).strip()
    if v_str == "":
        return default
    try:
        return int(v_str)
    except Exception:
        # 如果无法解析为 int，回退到默认，并记录
        print(f"[{now_str()}] WARNING: env {name} value '{v}' is not integer, using default {default}", flush=True)
        return default

MIN_BOARDS_FOR_PAW = safe_int_env("MIN_BOARDS_FOR_PAW", 3)   # 放水至少 3 张长龙（默认3）
MID_LONG_REQ = safe_int_env("MID_LONG_REQ", 2)              # 中等胜率长龙桌数（默认2）
COOLDOWN_MINUTES = safe_int_env("COOLDOWN_MINUTES", 10)     # 冷却分钟（默认10）

# image analysis params (tweakable)
HSV_RED_LOW1 = np.array([0, 100, 90])
HSV_RED_HIGH1 = np.array([10, 255, 255])
HSV_RED_LOW2 = np.array([160, 100, 90])
HSV_RED_HIGH2 = np.array([179, 255, 255])
HSV_BLUE_LOW = np.array([95, 60, 50])
HSV_BLUE_HIGH = np.array([140, 255, 255])

MIN_CONTOUR_AREA = 10

# ---------------- helpers ----------------
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置 (TG_BOT_TOKEN/TG_CHAT_ID). 跳过发送.")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=25)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log("Telegram 返回错误: " + str(j))
            return False
    except Exception as e:
        log("Telegram 发送异常: " + str(e))
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ---------------- image detection ----------------
def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def detect_points(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask_r = cv2.inRange(hsv, HSV_RED_LOW1, HSV_RED_HIGH1) | cv2.inRange(hsv, HSV_RED_LOW2, HSV_RED_HIGH2)
    mask_b = cv2.inRange(hsv, HSV_BLUE_LOW, HSV_BLUE_HIGH)
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    pts = []
    def extract(mask, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA: continue
            M = cv2.moments(cnt)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx, cy, label))
    extract(mask_r, 'B')
    extract(mask_b, 'P')
    return pts

def cluster_board_regions(points, w, h):
    if not points:
        return []
    coords = np.array([[p[0], p[1]] for p in points])
    # heuristic: try to cluster into up to 12-24 clusters depending on point count
    k = max(1, min(20, len(points)//6))
    try:
        km = KMeans(n_clusters=k, random_state=0).fit(coords)
        labs = km.labels_
        regions = []
        for lab in range(k):
            pts_lab = coords[labs==lab]
            if pts_lab.shape[0] < 3: continue
            x0,y0 = int(pts_lab[:,0].min()), int(pts_lab[:,1].min())
            x1,y1 = int(pts_lab[:,0].max()), int(pts_lab[:,1].max())
            # expand margin
            pad = 12
            rx = max(0, x0-pad); ry = max(0, y0-pad)
            rw = min(w - rx, (x1-x0)+pad*2); rh = min(h - ry, (y1-y0)+pad*2)
            regions.append((rx, ry, rw, rh))
        # if nothing or too few, fallback grid method
        if len(regions) < 1:
            # simple grid search for dense cells
            cell = max(60, min(w,h)//12)
            cols = math.ceil(w/cell); rows = math.ceil(h/cell)
            grid = [[0]*cols for _ in range(rows)]
            for x,y,_ in points:
                cx=int(x//cell); cy=int(y//cell)
                if 0<=cx<cols and 0<=cy<rows:
                    grid[cy][cx]+=1
            hits=[]
            thr=6
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c]>=thr:
                        hits.append((r,c))
            merged=[]
            for (r,c) in hits:
                x=c*cell; y=r*cell; w0=cell; h0=cell
                merged.append((x,y,w0,h0))
            regions = merged
        return regions
    except Exception as e:
        log("聚类异常:" + str(e))
        return []

def analyze_region(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    pts = detect_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "runs":[], "cols_info":[]}
    # cluster into columns by x coordinate
    xs = [p[0] for p in pts]
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for i in sorted_idx:
        xv = xs[i]
        placed=False
        for g in col_groups:
            # mean x of group
            mx = sum([pts[j][0] for j in g])/len(g)
            if abs(mx - xv) <= max(8, w//40):
                g.append(i); placed=True; break
        if not placed: col_groups.append([i])
    # for each col, sort by y
    cols_info=[]
    for g in col_groups:
        col_pts = sorted([pts[i] for i in g], key=lambda t: t[1])
        seq = [p[2] for p in col_pts]
        cols_info.append(seq)
    # order columns left->right
    # get column x positions for sorting
    col_xs=[]
    for g in col_groups:
        col_xs.append(sum([pts[i][0] for i in g])/len(g))
    order = sorted(range(len(col_xs)), key=lambda i: col_xs[i])
    seqs_ordered = [cols_info[i] for i in order]
    # flatten as bead-plate reading
    flattened=[]
    maxh = max((len(s) for s in seqs_ordered), default=0)
    for r in range(maxh):
        for col in seqs_ordered:
            if r < len(col):
                flattened.append(col[r])
    # compute runs
    runs=[]
    if flattened:
        cur = {"color":flattened[0], "len":1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":flattened[k], "len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # 多连/连珠 detection heuristic:
    # check if there exists a sequence of >=3 contiguous columns where each column has a top continuous run >=4
    col_top_runs=[]
    for col in seqs_ordered:
        # count top-run length
        if not col: col_top_runs.append(0)
        else:
            top = col[0]
            cnt = 1
            for i in range(1,len(col)):
                if col[i]==top: cnt+=1
                else: break
            col_top_runs.append(cnt)
    multi_chain_exists=False
    # find consecutive columns with top run >=4 of same color
    for i in range(0, max(0, len(col_top_runs)-2)):
        if col_top_runs[i] >=4 and col_top_runs[i+1] >=4 and col_top_runs[i+2] >=4:
            # ensure they are not alternating colors in top element:
            if seqs_ordered[i] and seqs_ordered[i+1] and seqs_ordered[i+2]:
                if seqs_ordered[i][0] == seqs_ordered[i+1][0] == seqs_ordered[i+2][0]:
                    multi_chain_exists = True
                    break
    # determine category
    category = "other"
    if maxRun >= 10: category="super_long"
    elif maxRun >= 8: category="long"
    elif maxRun >= 4: category="longish"
    elif maxRun == 1: category="single"
    return {"total": len(flattened), "maxRun": maxRun, "category": category, "runs": runs, "multi_chain": multi_chain_exists, "col_top_runs": col_top_runs}

# ---------------- classify overall ----------------
def classify_all(board_stats):
    longCount = sum(1 for b in board_stats if b["category"] in ("long","super_long"))
    superCount = sum(1 for b in board_stats if b["category"]=="super_long")
    # 放水条件：longCount >= MIN_BOARDS_FOR_PAW
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    # 中等胜率（中上）条件：
    # 至少 3 張桌子 each has multi_chain True AND global longCount >= MID_LONG_REQ
    multi_chain_boards = sum(1 for b in board_stats if b.get("multi_chain"))
    if multi_chain_boards >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount
    # 判断稀疏/收割
    sparse = sum(1 for b in board_stats if b["total"] < 6)
    n = max(1, len(board_stats))
    if sparse >= 0.6 * n:
        return "胜率调低 / 收割时段", longCount, superCount
    return "胜率中等（平台收割中等时段）", longCount, superCount

# ---------------- Playwright capture ----------------
def capture_screenshot_from_dg():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        context = browser.new_context(viewport={"width":1600,"height":900})
        page = context.new_page()
        for url in DG_LINKS:
            try:
                log(f"访问 {url}")
                page.goto(url, timeout=45000)
                time.sleep(2)
                # try sequence of interactions to get into the DG platform:
                # 1) try clicking text "Free" or localized alternatives
                possible_texts = ["Free", "免费试玩", "免费", "Play Free", "试玩", "免费试用"]
                for t in possible_texts:
                    try:
                        loc = page.locator(f"text={t}")
                        if loc.count() > 0:
                            loc.first.click(timeout=3000)
                            log(f"点击文字: {t}")
                            time.sleep(1.2)
                            break
                    except:
                        pass
                # 2) try scrolling and wheel to trigger security scroll-bar
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight/3)")
                    time.sleep(0.5)
                    page.evaluate("window.scrollTo(0, 0)")
                    time.sleep(0.5)
                    for _ in range(3):
                        page.mouse.wheel(0, 400)
                        time.sleep(0.4)
                except:
                    pass
                # wait a bit for table to load
                time.sleep(4)
                # finally screenshot the viewport (not necessarily full page)
                shot = page.screenshot(full_page=True)
                browser.close()
                return shot
            except Exception as e:
                log("访问/交互异常: " + str(e))
                try:
                    browser.close()
                except:
                    pass
                continue
    return None

# ---------------- main flow ----------------
def main():
    try:
        state = load_state()
        log("开始检测循环。")
        shot_bytes = capture_screenshot_from_dg()
        if not shot_bytes:
            log("未截到页面截图，本次结束。")
            return
        pil = Image.open(BytesIO(shot_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h,w = bgr.shape[:2]
        pts = detect_points(bgr)
        log(f"检测到点数: {len(pts)}")
        if len(pts) < MIN_POINTS_DETECTED:
            log("检测到点数太少，可能页面未成熟或识别阈值需调。保存并退出。")
            # save diagnostic
            debug = {"ts": now_str(), "points": len(pts)}
            with open(LAST_SUMMARY, "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
            return

        regions = cluster_board_regions(pts, w, h)
        log(f"聚类出候选桌数: {len(regions)}")
        board_stats=[]
        for i,reg in enumerate(regions):
            st = analyze_region(bgr, reg)
            st["region_idx"]=i+1
            board_stats.append(st)

        overall, longCount, superCount = classify_all(board_stats)
        log(f"判定 -> {overall} (长/超长龙桌数={longCount}, 超长龙={superCount})")

        # save last summary
        debug = {"ts": now_str(), "overall": overall, "longCount": longCount, "superCount": superCount, "boards": board_stats[:40]}
        with open(LAST_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)

        # state transition logic
        was_active = state.get("active", False)
        is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
        now_iso = datetime.now(TZ).isoformat()

        if is_active_now and not was_active:
            # start event
            start_time = now_iso
            # estimate end time based on historical durations (minutes)
            history = state.get("history", [])
            durations = [h.get("duration_minutes") for h in history if h.get("duration_minutes")]
            est_minutes = int(round(sum(durations)/len(durations))) if durations else 10
            est_end = (datetime.now(TZ) + timedelta(minutes=est_minutes)).strftime("%Y-%m-%d %H:%M:%S")
            emoji = "🚨"
            msg = f"{emoji} [DG提醒] {overall} 開始\n時間: {now_str()}\n長/超长龙桌數={longCount}，超长龙={superCount}\n估計結束: {est_end}，約 {est_minutes} 分鐘 (基於歷史)"
            send_telegram(msg)
            state["active"] = True
            state["kind"] = overall
            state["start_time"] = start_time
            state["last_seen"] = now_iso
            save_state(state)
            log("事件開始並已發送開始提醒。")
            return

        if is_active_now and was_active:
            # still active: update last seen
            state["last_seen"] = now_iso
            state["kind"] = overall
            save_state(state)
            log("事件仍在，已更新 last_seen。")
            return

        if (not is_active_now) and was_active:
            # event ended
            start = datetime.fromisoformat(state.get("start_time"))
            end = datetime.now(TZ)
            duration_m = int(round((end - start).total_seconds() / 60.0))
            history = state.get("history", [])
            history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_m})
            history = history[-120:]
            state_new = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history}
            save_state(state_new)
            emoji = "✅"
            msg = f"{emoji} [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end.isoformat()}\n實際持續: {duration_m} 分鐘"
            send_telegram(msg)
            log("事件已結束，發送結束通知並記錄歷史。")
            return

        # else: not active and was not active
        save_state(state)
        log("目前非放水/中上時段，不發提醒。")
    except Exception as e:
        log("主流程異常: " + str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()
