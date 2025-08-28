# -*- coding: utf-8 -*-
"""
改进版 main.py — 针对你贴的日志问题修正：
- 更强的进入实盘检测（等待彩点阈值或重试）
- 支持 popup/frame 查找与滑块处理
- 更精确实现“中等胜率（中上）”判定规则（连续3列多连 + >=2 张长龙）
- 更详尽日志与 last_run_summary.json 输出
"""
import os, time, json, math, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO

import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from sklearn.cluster import KMeans

# ---------------- CONFIG ----------------
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID  = os.environ.get("TG_CHAT_ID", "").strip()
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]

MIN_POINTS_FOR_REAL_BOARD = int(os.environ.get("MIN_POINTS_FOR_REAL_BOARD", "40"))  # 彩点阈值，判断是否已进入实盘
MAX_WAIT_SECONDS = int(os.environ.get("MAX_WAIT_SECONDS", "30"))  # 等待实盘加载最大秒数（单次重试）
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "2"))  # 失败后重试次数
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

# ---------------- helpers ----------------
def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置，跳过发送")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id":TG_CHAT_ID, "text": text, "parse_mode":"HTML"}, timeout=15)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功")
            return True
        else:
            log(f"Telegram 返回: {j}")
            return False
    except Exception as e:
        log(f"Telegram 发送异常: {e}")
        return False

def load_state():
    if not Path(STATE_FILE).exists():
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))

def save_state(s):
    Path(STATE_FILE).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- image utilities ----------------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_points(bgr):
    """HSV 检测红/蓝点，返回 list of (x,y,label)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0,100,80]), np.array([10,255,255])
    lower2, upper2 = np.array([160,100,80]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb, upperb = np.array([90,60,50]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)

    pts=[]
    for mask,label in [(mask_r,'B'), (mask_b,'P')]:
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < 10: continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            pts.append((cx,cy,label))
    # fallback: HoughCircles on gray to find circles (if hsv misses)
    if len(pts) < 8:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12, param1=50, param2=18, minRadius=4, maxRadius=18)
        if circles is not None:
            circles = np.round(circles[0,:]).astype("int")
            for (x,y,r) in circles:
                # sample color to guess label
                px = max(0, min(bgr.shape[1]-1, x)); py = max(0, min(bgr.shape[0]-1, y))
                b,g,r0 = bgr[py,px]
                if r0 > b+30: label='B'
                elif b > r0+30: label='P'
                else: label='B'
                pts.append((x,y,label))
    return pts

# ---------------- clustering into regions (tables) ----------------
def cluster_regions(points, img_w, img_h):
    if not points: return []
    cell = max(60, int(min(img_w,img_h)/12))
    cols = math.ceil(img_w / cell); rows = math.ceil(img_h/cell)
    grid = [[0]*cols for _ in range(rows)]
    for x,y,_ in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx]+=1
    hits=[(r,c) for r in range(rows) for c in range(cols) if grid[r][c]>=6]
    if not hits:
        pts_arr = np.array([[p[0],p[1]] for p in points])
        k = min(8, max(1, len(points)//8))
        try:
            km = KMeans(n_clusters=k, random_state=0).fit(pts_arr)
            regs=[]
            for lab in range(k):
                sel = pts_arr[km.labels_==lab]
                if sel.shape[0]==0: continue
                x0,y0 = sel.min(axis=0); x1,y1 = sel.max(axis=0)
                regs.append((int(max(0,x0-10)), int(max(0,y0-10)), int(min(img_w,x1-x0+20)), int(min(img_h,y1-y0+20))))
            return regs
        except Exception:
            return []
    # merge
    rects=[]
    for (r,c) in hits:
        x = c*cell; y = r*cell; w=cell; h=cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x>rx+rw+cell or x+w<rx-cell or y>ry+rh+cell or y+h<ry-cell):
                nx=min(rx,x); ny=min(ry,y); nw=max(rx+rw, x+w)-nx; nh=max(ry+rh, y+h)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged: rects.append((x,y,w,h))
    regs=[]
    for (x,y,w,h) in rects:
        nx=max(0,x-10); ny=max(0,y-10); nw=min(img_w-nx, w+20); nh=min(img_h-ny, h+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ---------------- analyze single region (table) ----------------
def analyze_region(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    pts = detect_points(crop)
    # map points into columns by x clustering
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "runs":[], "multiRuns":0, "cols_max":[]}
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in pts_sorted]
    col_groups=[]
    for i,xv in enumerate(xs):
        placed=False
        for grp in col_groups:
            gx = np.mean([pts_sorted[j][0] for j in grp])
            if abs(gx - xv) <= max(8, w//45):
                grp.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    sequences=[]
    cols_max=[]
    for grp in col_groups:
        col_pts = sorted([pts_sorted[i] for i in grp], key=lambda t:t[1])
        seq = [p[2] for p in col_pts]
        sequences.append(seq)
        # per-column max run
        m=0
        if seq:
            cur=seq[0]; ln=1
            for s in seq[1:]:
                if s==cur: ln+=1
                else:
                    m=max(m,ln); cur=s; ln=1
            m=max(m,ln)
        cols_max.append(m)
    # flattened by rows top->bottom per column
    flattened=[]
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
    # runs overall
    runs=[]
    if flattened:
        cur=flattened[0]; ln=1
        for t in flattened[1:]:
            if t==cur: ln+=1
            else:
                runs.append({"color":cur,"len":ln}); cur=t; ln=1
        runs.append({"color":cur,"len":ln})
    maxRun = max((r["len"] for r in runs), default=0)
    cat="other"
    if maxRun>=10: cat="super_long"
    elif maxRun>=8: cat="long"
    elif maxRun>=4: cat="longish"
    elif maxRun==1: cat="single"
    # multiRuns: total count of runs with len>=4
    multiRuns = sum(1 for r in runs if r["len"]>=4)
    # new: detect "连续3列多连" -> check cols_max for >=4 in at least 3 adjacent columns
    consec3=False
    if len(cols_max) >= 3:
        for i in range(len(cols_max)-2):
            if cols_max[i] >=4 and cols_max[i+1] >=4 and cols_max[i+2] >=4:
                consec3=True; break
    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "runs":runs, "multiRuns":multiRuns, "cols_max":cols_max, "consec3":consec3}

# ---------------- classify overall per your strict rules ----------------
def classify_overall(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    # per-table consec3 count:
    consec3_count = sum(1 for b in board_stats if b.get('consec3', False))
    # requirement for 中等勝率(中上): 有 3 張桌子滿足連續3列多連(consec3) AND 有 2 張桌子為 長龍/超長龍 (可以同桌)
    if consec3_count >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount
    # 放水時段：至少 MIN_BOARDS_FOR_PAW 張桌為 長龍/超龍
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    # sparse -> 收割
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if len(board_stats)>0 and sparse >= len(board_stats)*0.6:
        return "胜率调低 / 收割时段", longCount, superCount
    return "胜率中等（平台收割中等时段）", longCount, superCount

# ---------------- Playwright: open page, click Free, handle popup/frame and slider ----------------
def try_enter_game(play, url):
    """Open url, attempt to click Free, handle popup/frames and slider, wait until real board detected."""
    browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled"])
    try:
        context = browser.new_context(viewport={"width":1366,"height":768})
        context.add_init_script("() => { Object.defineProperty(navigator, 'webdriver', {get: () => false}); }")
        page = context.new_page()
        page.set_default_timeout(20000)
        log(f"打开 {url}")
        page.goto(url, timeout=20000)
        time.sleep(1.2)
        # try click Free/免费/Play Free with retries
        clicked=False
        texts = ["Free","免费试玩","免费","Play Free","试玩","进入","Play"]
        for txt in texts:
            try:
                els = page.locator(f"text={txt}")
                if els.count() > 0:
                    try:
                        els.first.click(timeout=3000)
                        clicked=True
                        log(f"点击按钮文字: {txt}")
                        break
                    except Exception:
                        try:
                            page.evaluate("(e)=>e.click()", els.first)
                            clicked=True; log(f"JS 点击: {txt}"); break
                        except Exception:
                            pass
            except Exception:
                continue
        if not clicked:
            # try first big button
            try:
                btn = page.query_selector("button")
                if btn:
                    btn.click(timeout=2000); clicked=True; log("点击第一个 button (fallback)")
            except Exception:
                pass

        # after clicking, check for new pages (popups)
        time.sleep(0.8)
        pages = context.pages
        target_page = None
        if len(pages) > 1:
            target_page = pages[-1]; log("发现新页面，切换到新页面")
        else:
            target_page = page

        # attempt slider within target_page and in frames
        def attempt_slider_on(p):
            try:
                # look for range inputs or role=slider
                el = p.query_selector("input[type=range]")
                if el:
                    bb = el.bounding_box(); 
                    if bb:
                        sx=bb["x"]+5; sy=bb["y"]+bb["height"]/2; ex=bb["x"]+bb["width"]-6
                        p.mouse.move(sx,sy); p.mouse.down(); p.mouse.move(ex,sy,steps=28); p.mouse.up()
                        log("slider input 拖动成功"); return True
                # role=slider
                els = p.locator("[role=slider]")
                if els.count()>0:
                    bb = els.first.bounding_box()
                    if bb:
                        sx=bb["x"]+4; sy=bb["y"]+bb["height"]/2; ex=bb["x"]+bb["width"]-4
                        p.mouse.move(sx,sy); p.mouse.down(); p.mouse.move(ex,sy,steps=30); p.mouse.up()
                        log("role=slider 拖动成功"); return True
                # class name includes slide/drag/slider
                el = p.query_selector("[class*=slide], [class*=drag], [class*=slider]")
                if el:
                    bb=el.bounding_box()
                    if bb:
                        sx=bb["x"]+5; sy=bb["y"]+bb["height"]/2; ex=bb["x"]+bb["width"]-5
                        p.mouse.move(sx,sy); p.mouse.down(); p.mouse.move(ex,sy,steps=30); p.mouse.up()
                        log("class slide/drag 拖动成功"); return True
                # fallback generic drag on visible area center
                vp = p.viewport_size
                if vp:
                    sx=vp['width']*0.25; sy=vp['height']*0.6; ex=vp['width']*0.75
                    p.mouse.move(sx,sy); p.mouse.down(); p.mouse.move(ex,sy,steps=30); p.mouse.up()
                    log("通用区域拖动尝试")
                    return True
            except Exception as e:
                log(f"attempt_slider_on 异常: {e}")
            return False

        # attempt on main page and frames
        try:
            attempt_slider_on(target_page)
            for f in target_page.frames:
                try:
                    attempt_slider_on(f)
                except Exception:
                    pass
        except Exception as e:
            log(f"滑块整体尝试异常: {e}")

        # now wait until we detect enough colored points (real board) with timeout
        start = time.time()
        while time.time() - start < MAX_WAIT_SECONDS:
            try:
                # get screenshot of target page (full)
                shot = target_page.screenshot(full_page=True)
                pil = Image.open(BytesIO(shot)).convert("RGB")
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                pts = detect_points(bgr)
                log(f"等待检测中，当前彩点数: {len(pts)} (阈值 {MIN_POINTS_FOR_REAL_BOARD})")
                if len(pts) >= MIN_POINTS_FOR_REAL_BOARD:
                    log("检测到足够彩点，视为已进入实盘")
                    return shot
            except Exception as e:
                log(f"等待检测异常: {e}")
            time.sleep(1.2)
        # 若超时仍未达到阈值，返回最后截图（可能是登录页）
        try:
            shot = target_page.screenshot(full_page=True)
            return shot
        except Exception:
            return None
    finally:
        try:
            browser.close()
        except:
            pass

# ---------------- main flow ----------------
def main():
    log("开始一次检测循环")
    state = load_state()
    screenshot = None
    # try each link with retries
    with sync_playwright() as p:
        for url in DG_LINKS:
            ok=False
            for attempt in range(RETRY_ATTEMPTS+1):
                try:
                    shot = try_enter_game(p, url)
                    if shot:
                        screenshot = shot; ok=True; break
                except Exception as e:
                    log(f"访问 {url} 第 {attempt+1} 次异常: {e}")
                time.sleep(1.0 + attempt*0.5)
            if ok: break

    if not screenshot:
        log("未能取得截图，结束本次 run")
        save_state(state)
        return

    pil = pil_from_bytes(screenshot) if isinstance(screenshot, bytes) else Image.open(BytesIO(screenshot)).convert("RGB")
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    pts = detect_points(bgr)
    log(f"本次截图检测到彩点: {len(pts)}")
    if len(pts) < 8:
        log("彩点太少，可能仍未进入实盘，保存摘要并结束")
        summary = {"ts": nowstr(), "points":len(pts), "boards":[]}
        Path(LAST_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        save_state(state)
        return

    regions = cluster_regions(pts, w, h)
    log(f"聚类得到候选桌数: {len(regions)}")
    boards=[]
    for i,r in enumerate(regions):
        st = analyze_region(bgr, r)
        st['region_idx']=i+1; st['bbox']=r
        boards.append(st)
        log(f"桌 {i+1} -> total {st['total']} maxRun {st['maxRun']} cat {st['category']} consec3 {st.get('consec3')} cols_max {st.get('cols_max')[:6]}")

    overall, longCount, superCount = classify_overall(boards)
    log(f"判定: {overall} (长/超长龙={longCount}, 超={superCount}, 连续3列多连桌数={sum(1 for b in boards if b.get('consec3'))})")

    now_iso = datetime.now(TZ).isoformat()
    was_active = state.get("active", False)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    if is_active_now and not was_active:
        # start event
        history = state.get("history", [])
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        est = round(sum(durations)/len(durations)) if durations else 10
        est_end = (datetime.now(TZ) + timedelta(minutes=est)).strftime("%Y-%m-%d %H:%M")
        emoji = "🚩"
        msg = f"{emoji} <b>DG 提醒 — {overall}</b>\n偵測時間 (MYT): {now_iso}\n長/超长龙桌數={longCount}，超长龙={superCount}\n估計結束時間: {est_end}（約 {est} 分鐘）\n請手動確認實況後入場。"
        send_telegram(msg)
        state = {"active":True, "kind":overall, "start_time":now_iso, "last_seen":now_iso, "history":state.get("history", [])}
        save_state(state)
        log("觸發開始通知並保存狀態")
    elif is_active_now and was_active:
        state["last_seen"] = now_iso; state["kind"]=overall; save_state(state); log("事件持續，更新 last_seen")
    elif (not is_active_now) and was_active:
        # ended
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ); dur_min = round((end-start).total_seconds()/60)
        entry = {"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes":dur_min}
        hist = state.get("history", []); hist.append(entry); hist = hist[-120:]
        new_state = {"active":False, "kind":None, "start_time":None, "last_seen":None, "history":hist}
        save_state(new_state)
        emoji = "✅"
        msg = f"{emoji} <b>DG 提醒 — {state.get('kind')} 已結束</b>\n開始: {entry['start_time']}\n結束: {entry['end_time']}\n實際持續: {dur_min} 分鐘"
        send_telegram(msg)
        log("事件結束，已發送結束通知")
    else:
        save_state(state); log("非事件時段，不發提醒")

    # write last summary
    summary = {"ts": nowstr(), "overall":overall, "longCount":longCount, "superCount":superCount, "points":len(pts), "boards":boards[:40]}
    Path(LAST_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("已寫入 last_run_summary.json")
    return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"主程式例外: {e}")
        raise
