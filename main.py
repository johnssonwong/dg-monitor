# -*- coding: utf-8 -*-
"""
DG 实盘检测主脚本（针对 GitHub Actions / Playwright）
功能：
- 使用 Playwright 自动打开 DG 链接 (dg18.co/wap 或 dg18.co)
- 点击 "Free" / "免费试玩" 并尝试通过弹窗/滑块安全条（自动拖动）
- 截图 DG 实盘页面并用 OpenCV 分析每个桌面：识别红/蓝珠、计算连长、判断多连/长龙/超长龙、以及判断整体时段
- 按规则判断：放水时段 / 中等胜率（中上） / 胜率中等 / 胜率调低（收割）
- 仅在放水 或 中等胜率（中上） 触发 Telegram 提醒（开始通知带 emoji、估算结束时间）；当该局势结束时再发结束通知（并写入历史以便估算将来持续时间）
- 将状态存储在 state.json（并由 workflow commit 回 repo），以作为历史数据与去重/冷却依据
注意：请在 GitHub Secrets 中放入 TG_BOT_TOKEN 与 TG_CHAT_ID
"""

import os, time, json, math, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

# Playwright
from playwright.sync_api import sync_playwright

# sklearn kmeans for fallback clustering
from sklearn.cluster import KMeans

# ------------------ 配置（可通过 env / secrets 覆盖） ------------------
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID  = os.environ.get("TG_CHAT_ID", "").strip()

DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# 判定参数（必要时可在 workflow 环境变量中覆盖）
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))   # 放水至少满足桌数（≥3）
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))              # 中等胜率需要 >=2 张长龙/超长龙
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))     # 在事件开始后将进入冷却，避免重复开始提醒

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"

# 马来西亚时区
TZ = timezone(timedelta(hours=8))

# ------------------ logging ------------------
def log(msg):
    print(f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ------------------ Telegram ------------------
def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 配置缺失，无法发送消息。")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram: 已发送通知。")
            return True
        else:
            log(f"Telegram API 返回错误: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 时异常: {e}")
        return False

# ------------------ state ------------------
def load_state():
    if not Path(STATE_FILE).exists():
        s = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
        return s
    return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))

def save_state(state):
    Path(STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------ 图像处理: 检测红/蓝圆点 ------------------
def pil_from_bytes(data):
    return Image.open(BytesIO(data)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_points(bgr):
    """
    更稳健的 HSV 阈值检测红/蓝圆点。返回 points list: (x,y,label) label 'B' (red 庄) or 'P' (blue 闲)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # red range (two ranges)
    lower1 = np.array([0, 100, 80]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([160,100,80]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb = np.array([90, 60, 50]); upperb = np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)

    # morphology to reduce noise
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)

    points = []
    for mask,label in [(mask_r,'B'), (mask_b,'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 12:  # skip noise; 可调整
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            points.append((cx, cy, label))
    return points

# ------------------ 聚类为桌子区域（启发式） ------------------
def cluster_to_regions(points, img_w, img_h):
    if not points:
        return []
    # coarse grid based clustering
    cell = max(64, int(min(img_w, img_h)/12))
    cols = math.ceil(img_w / cell); rows = math.ceil(img_h / cell)
    grid_counts = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x // cell)
        cy = min(rows-1, y // cell)
        grid_counts[cy][cx] += 1
    # find high-density cells
    thr = 6
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid_counts[r][c] >= thr]
    if not hits:
        # fallback: kmeans on points positions
        pts = np.array([[p[0], p[1]] for p in points])
        k = min(8, max(1, len(points)//8))
        try:
            km = KMeans(n_clusters=k, random_state=0).fit(pts)
            regs = []
            for lab in range(k):
                sel = pts[km.labels_==lab]
                if sel.shape[0]==0: continue
                x0,y0 = sel.min(axis=0); x1,y1 = sel.max(axis=0)
                regs.append((int(max(0,x0-10)), int(max(0,y0-10)), int(min(img_w,x1-x0+20)), int(min(img_h,y1-y0+20))))
            return regs
        except Exception:
            return []
    # merge hits to rectangles
    rects = []
    for (r,c) in hits:
        x = c*cell; y = r*cell; w = cell; h = cell
        merged = False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                nx = min(rx,x); ny = min(ry,y)
                nw = max(rx+rw, x+w) - nx
                nh = max(ry+rh, y+h) - ny
                rects[i] = (nx,ny,nw,nh)
                merged = True
                break
        if not merged:
            rects.append((x,y,w,h))
    # expand slightly
    regs = []
    for (x,y,w,h) in rects:
        nx = max(0, x-10); ny = max(0, y-10)
        nw = min(img_w-nx, w+20); nh = min(img_h-ny, h+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ------------------ 分析单个桌子 ------------------
def analyze_region(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    pts = detect_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "runs":[]}
    # group by approximate column using x coordinate clustering
    xs = [p[0] for p in pts]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups = []
    for i in order:
        xi = xs[i]
        placed=False
        for grp in col_groups:
            gx = np.mean([pts[j][0] for j in grp])
            if abs(gx - xi) <= max(8, w//45):
                grp.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    # for each column, sort by y and produce sequence of colors
    sequences = []
    for grp in col_groups:
        col_pts = sorted([pts[i] for i in grp], key=lambda t: t[1])
        seq = [p[2] for p in col_pts]
        sequences.append(seq)
    # flatten columns: top-down in each column, left->right across columns
    flattened = []
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
    # compute runs
    runs = []
    if flattened:
        cur_color = flattened[0]; cur_len = 1
        for k in range(1,len(flattened)):
            if flattened[k] == cur_color:
                cur_len += 1
            else:
                runs.append({"color":cur_color, "len":cur_len})
                cur_color = flattened[k]; cur_len = 1
        runs.append({"color":cur_color, "len":cur_len})
    maxRun = max((r["len"] for r in runs), default=0)
    # detect 多连/连珠: count of runs with len>=4
    multi_runs = sum(1 for r in runs if r["len"] >= 4)
    # classify
    category = "other"
    if maxRun >= 10: category = "super_long"
    elif maxRun >= 8: category = "long"
    elif maxRun >= 4: category = "longish"
    elif maxRun == 1: category = "single"
    return {"total":len(flattened), "maxRun":maxRun, "category":category, "runs":runs, "multiRuns":multi_runs}

# ------------------ Playwright: 打开页面并进入实盘 ------------------
def capture_screenshot_from_dg(play, url, timeout_total=40):
    """
    打开 url，尝试点击 Free / 免费试玩；处理弹窗并拖动安全滑块（若存在）；等待实盘加载后截屏
    返回截图 bytes 或 None
    """
    browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"])
    try:
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36", viewport={"width":1366, "height":768}, locale="en-US")
        # reduce navigator.webdriver
        context.add_init_script("() => { Object.defineProperty(navigator, 'webdriver', {get: () => false}); }")
        page = context.new_page()
        log(f"访问: {url}")
        page.goto(url, timeout=20000)
        time.sleep(1.2)
        # try several times to click Free / 免费试玩 or Free text
        clicked=False
        for txt in ["Free", "免费试玩", "免费", "Play Free", "试玩", "进入"]:
            try:
                loc = page.locator(f"text={txt}")
                if loc.count() > 0:
                    try:
                        loc.first.click(timeout=3000)
                        clicked=True
                        log(f"点击文本按钮: {txt}")
                        break
                    except Exception:
                        # try JS click
                        page.evaluate("(el) => el.click()", loc.first)
                        clicked = True
                        break
            except Exception:
                continue
        if not clicked:
            # try to click common button elements
            try:
                btn = page.query_selector("button")
                if btn:
                    btn.click(timeout=2000); clicked=True; log("尝试点击第一个 button")
            except Exception:
                pass

        # give time for popup to appear
        time.sleep(1.2)
        # handle slider-like element: attempt known selectors; otherwise attempt to drag an element near center
        try:
            # attempt to find input[type=range] or role=slider
            slider = None
            try:
                slider = page.query_selector("input[type=range]")
            except:
                slider = None
            if not slider:
                try:
                    slider = page.locator("[role=slider]").first
                    if slider.count()==0: slider=None
                except:
                    slider=None
            if slider:
                bb = slider.bounding_box()
                if bb:
                    sx = bb["x"] + 5; sy = bb["y"] + bb["height"]/2
                    ex = bb["x"] + bb["width"] - 5
                    page.mouse.move(sx, sy); page.mouse.down(); page.mouse.move(ex, sy, steps=20); page.mouse.up()
                    log("找到 slider，完成拖动")
            else:
                # generic drag attempt: find element with class contains 'slide' or 'drag'
                try:
                    el = page.query_selector("[class*=slide], [class*=drag], [class*=slider]")
                    if el:
                        bb = el.bounding_box()
                        if bb:
                            sx = bb["x"] + 5; sy = bb["y"] + bb["height"]/2
                            ex = bb["x"] + bb["width"] - 5
                            page.mouse.move(sx, sy); page.mouse.down(); page.mouse.move(ex, sy, steps=25); page.mouse.up()
                            log("尝试拖动通用滑块元素")
                    else:
                        # fallback: attempt to drag a visible small rectangle near center bottom
                        wv = page.viewport_size
                        if wv:
                            sx = wv['width']*0.25; sy = wv['height']*0.6
                            ex = wv['width']*0.75
                            page.mouse.move(sx, sy); page.mouse.down(); page.mouse.move(ex, sy, steps=25); page.mouse.up()
                            log("尝试通用区域滑动（fallback）")
                except Exception as e:
                    log(f"滑块拖动尝试失败: {e}")
        except Exception as e:
            log(f"滑块逻辑异常: {e}")

        # allow time for game panel to load
        time.sleep(4)
        # Try to detect presence of game grids: look for many colored circles via JS -> take screenshot
        shot = page.screenshot(full_page=True)
        log("已截图并返回")
        try:
            context.close()
        except:
            pass
        return shot
    finally:
        try:
            browser.close()
        except:
            pass

# ------------------ 依据每桌统计做整体判定 ------------------
def classify_overall(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    multi_count = sum(1 for b in board_stats if b.get('multiRuns',0) >= 3)  # 每桌至少3个多连（>=4）的 run
    longish_count = sum(1 for b in board_stats if b['category']=='longish')
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    n = len(board_stats)
    # 放水时段：至少 MIN_BOARDS_FOR_PAW 张为 长龙/超长龙（>=8）
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    # 中等胜率（中上）: 至少 3 张桌子满足“连续3排多连/连珠”（我们定义为单桌 multiRuns>=3），并且至少 2 张桌子为长龙/超长龙（>=8）
    cond1 = multi_count >= 3
    cond2 = longCount >= MID_LONG_REQ
    if cond1 and cond2:
        return "中等胜率（中上）", longCount, superCount
    # 若多数桌空旷 -> 收割
    if n>0 and sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount
    return "胜率中等（平台收割中等时段）", longCount, superCount

# ------------------ 主流程 ------------------
def main():
    global TG_BOT_TOKEN, TG_CHAT_ID
    log("开始检测循环...")
    state = load_state()
    screenshot = None
    with sync_playwright() as p:
        # try both links with retries
        for url in DG_LINKS:
            try:
                for attempt in range(2):
                    shot = capture_screenshot_from_dg(p, url)
                    if shot:
                        screenshot = shot
                        break
                    time.sleep(1.5)
                if screenshot: break
            except Exception as e:
                log(f"访问 {url} 时失败: {e}")
                continue

    if not screenshot:
        log("无法获得实盘截图，本次结束。")
        save_state(state)
        return

    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    points = detect_points(bgr)
    log(f"检测到彩点数量: {len(points)}")
    if len(points) < 8:
        log("彩点偏少（可能界面未进入实盘或识别门槛），本次不判定。")
        save_state(state)
        return

    regions = cluster_to_regions(points, w, h)
    log(f"聚类出候选小桌: {len(regions)}")
    board_stats = []
    for idx, reg in enumerate(regions):
        st = analyze_region(bgr, reg)
        st['region_idx'] = idx+1
        board_stats.append(st)

    overall, longCount, superCount = classify_overall(board_stats)
    log(f"本次判定 => {overall} （长/超长龙桌数={longCount}，超长龙={superCount}）")

    now_iso = datetime.now(TZ).isoformat()
    was_active = state.get("active", False)
    was_kind = state.get("kind", None)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上)".replace(")","")) or overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    # normalize flag: check two specific strings
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    # state transitions
    if is_active_now and not was_active:
        # start new event
        history = state.get("history", [])
        # estimate duration from history mean
        if history:
            durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
            est_minutes = round(sum(durations)/len(durations)) if durations else 10
        else:
            est_minutes = 10
        est_end_dt = datetime.now(TZ) + timedelta(minutes=est_minutes)
        est_end_str = est_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🚩"
        msg = f"{emoji} <b>DG 提醒 — {overall}</b>\n偵測時間 (MYT): {now_iso}\n長/超长龙桌數={longCount}，超长龙={superCount}\n估計結束時間: {est_end_str} （約 {est_minutes} 分鐘）\n\n提醒：此為系統實時偵測，請手動進場確認實況。"
        send_telegram(msg)
        # update state
        state = {"active":True, "kind":overall, "start_time": now_iso, "last_seen": now_iso, "history": state.get("history", [])}
        save_state(state)
        log("開始事件已記錄並發送 Telegram（若配置）。")
    elif is_active_now and was_active:
        state["last_seen"] = now_iso
        state["kind"] = overall
        save_state(state)
        log("事件持續中，更新 last_seen。")
    elif (not is_active_now) and was_active:
        # ended
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ)
        duration_min = round((end - start).total_seconds()/60)
        entry = {"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_min}
        hist = state.get("history", [])
        hist.append(entry)
        hist = hist[-120:]
        new_state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": hist}
        save_state(new_state)
        emoji = "✅"
        msg = f"{emoji} <b>DG 提醒 — {state.get('kind')} 已結束</b>\n開始: {entry['start_time']}\n結束: {entry['end_time']}\n實際持續: {duration_min} 分鐘"
        send_telegram(msg)
        log("事件結束已發送通知並保存歷史。")
    else:
        # not active, do nothing
        save_state(state)
        log("目前未處於放水/中上時段，不發提醒。")

    # write last_run_summary for debugging
    debug = {"ts": now_iso, "overall": overall, "longCount": longCount, "superCount": superCount, "boards": board_stats[:50]}
    Path(LAST_SUMMARY).write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    log("本次摘要已寫入 last_run_summary.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"主程式例外: {e}")
        raise
