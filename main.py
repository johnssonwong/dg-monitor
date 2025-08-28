# -*- coding: utf-8 -*-
"""
DG 监测脚本（最终版）
- 优先视觉识别（Playwright 截图 + OpenCV）
- 若无法进入实盘或截图点数不足 -> 自动退回网络模式 (捕获页面所有 XHR/Fetch JSON) 并解析真实牌面数据进行判定
- 完全使用你要求的判定规则（放水 / 中等胜率（中上） / 胜率中等 / 收割）
- Telegram 通知：仅在 放水 或 中等胜率（中上） 时开始通知，结束时发结束通知（含真实持续时间）
- 输出 last_run_summary.json 便于调试
"""
import os, time, json, math
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

# Visual thresholds
MIN_POINTS_FOR_REAL_BOARD = int(os.environ.get("MIN_POINTS_FOR_REAL_BOARD", "40"))
MAX_WAIT_SECONDS = int(os.environ.get("MAX_WAIT_SECONDS", "30"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "2"))

# Logic thresholds
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

# ---------------- helpers ----------------
def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置，跳过发送")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                          data={"chat_id":TG_CHAT_ID, "text":text, "parse_mode":"HTML"}, timeout=15)
        j = r.json()
        if j.get("ok"):
            log("Telegram 已发送")
            return True
        else:
            log(f"Telegram 返回: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
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
    """HSV + Hough fallback 圆点检测"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # red
    lower1,upper1 = np.array([0,100,80]), np.array([10,255,255])
    lower2,upper2 = np.array([160,100,80]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb,upperb = np.array([90,60,50]), np.array([140,255,255])
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
            if M['m00']==0: continue
            cx = int(M['m10']/M['m00']); cy=int(M['m01']/M['m00'])
            pts.append((cx,cy,label))
    # fallback Hough if too few
    if len(pts) < 8:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12, param1=50, param2=18, minRadius=4, maxRadius=18)
        if circles is not None:
            for (x,y,r) in np.round(circles[0,:]).astype("int"):
                px,py = max(0,min(bgr.shape[1]-1,x)), max(0,min(bgr.shape[0]-1,y))
                b,g,rr = bgr[py,px]
                if rr > b+20: lab='B'
                elif b > rr+20: lab='P'
                else: lab='B'
                pts.append((int(x),int(y),lab))
    return pts

# ---------------- clustering and per-table analysis ----------------
def cluster_regions(points, img_w, img_h):
    if not points: return []
    cell = max(60, int(min(img_w,img_h)/12))
    cols = math.ceil(img_w/cell); rows = math.ceil(img_h/cell)
    grid=[[0]*cols for _ in range(rows)]
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
    rects=[]
    for (r,c) in hits:
        x=r*c  # placeholder not used
    # merge adjacent
    rects=[]
    for (r,c) in hits:
        x = c*cell; y=r*cell; w=cell; h=cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x>rx+rw+cell or x+w<rx-cell or y>ry+rh+cell or y+h<ry-cell):
                nx=min(rx,x); ny=min(ry,y); nw=max(rx+rw, x+w)-nx; nh=max(ry+rh, y+h)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged:
            rects.append((x,y,w,h))
    regs=[]
    for (x,y,w,h) in rects:
        nx=max(0,x-10); ny=max(0,y-10); nw=min(img_w-nx,w+20); nh=min(img_h-ny,h+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_region(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    pts = detect_points(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","runs":[],"multiRuns":0,"cols_max":[],"consec3":False}
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in pts_sorted]
    col_groups=[]
    for i,xv in enumerate(xs):
        placed=False
        for grp in col_groups:
            gx = sum(pts_sorted[j][0] for j in grp)/len(grp)
            if abs(gx - xv) <= max(8, w//45):
                grp.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    sequences=[]; cols_max=[]
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
    flattened=[]; maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
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
    multiRuns = sum(1 for r in runs if r["len"]>=4)
    consec3=False
    if len(cols_max) >= 3:
        for i in range(len(cols_max)-2):
            if cols_max[i] >=4 and cols_max[i+1] >=4 and cols_max[i+2] >=4:
                consec3=True; break
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"runs":runs,"multiRuns":multiRuns,"cols_max":cols_max,"consec3":consec3}

def classify_overall(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    consec3_count = sum(1 for b in board_stats if b.get('consec3',False))
    # 中等胜率（中上） first: >=3 tables have consec3 AND >=2 tables long/ultra
    if consec3_count >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount
    # 放水： >= MIN_BOARDS_FOR_PAW longCount
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if len(board_stats)>0 and sparse >= len(board_stats)*0.6:
        return "胜率调低 / 收割时段", longCount, superCount
    return "胜率中等（平台收割中等时段）", longCount, superCount

# ---------------- Playwright with network capture ----------------
def try_visit_and_capture(play, url):
    """
    尝试打开 url, 点击 Free, 模拟滑动, 同时捕获 network responses (xhr/fetch)
    返回 dict:
      {"mode":"visual","screenshot":bytes, "points":int}
      或 {"mode":"network","api_candidates":[...parsed...]}
      或 None
    """
    browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"])
    try:
        context = browser.new_context(viewport={"width":1366,"height":768})
        # reduce webdriver flag
        context.add_init_script("() => { Object.defineProperty(navigator, 'webdriver', {get: () => false}); }")
        responses = []
        def on_response(resp):
            try:
                ct = resp.headers.get("content-type","")
                if resp.request.resource_type in ("xhr","fetch") or "json" in ct or "/api/" in resp.url:
                    # read body safely
                    try:
                        text = resp.text()
                        responses.append({"url":resp.url, "status":resp.status, "text": text})
                    except Exception:
                        pass
            except Exception:
                pass
        context.on("response", on_response)
        page = context.new_page()
        page.set_default_timeout(20000)
        log(f"打开 {url}")
        page.goto(url, timeout=20000)
        time.sleep(1.0)
        # attempt click text Free variants
        clicked=False
        for txt in ["Free","免费试玩","免费","Play Free","试玩","进入","Play"]:
            try:
                els = page.locator(f"text={txt}")
                if els.count() > 0:
                    try:
                        els.first.click(timeout=3000); clicked=True; log(f"点击 {txt}"); break
                    except Exception:
                        try:
                            page.evaluate("(e)=>e.click()", els.first); clicked=True; log(f"JS 点击 {txt}"); break
                        except Exception:
                            pass
            except Exception:
                continue
        if not clicked:
            try:
                btn = page.query_selector("button")
                if btn:
                    btn.click(timeout=2000); clicked=True; log("点击第一个 button fallback")
            except Exception:
                pass
        time.sleep(0.8)
        # try drag slider on page (simple)
        try:
            vp = page.viewport_size or {"width":1366,"height":768}
            sx = vp['width']*0.25; sy = vp['height']*0.6; ex = vp['width']*0.75
            page.mouse.move(sx, sy); page.mouse.down(); page.mouse.move(ex, sy, steps=30); page.mouse.up()
            log("页面层级拖动尝试")
        except Exception as e:
            log(f"拖动尝试异常: {e}")

        # wait loop: try until we see MIN_POINTS_FOR_REAL_BOARD or timeout
        start = time.time()
        last_shot = None
        while time.time() - start < MAX_WAIT_SECONDS:
            try:
                shot = page.screenshot(full_page=True)
                last_shot = shot
                pil = pil_from_bytes(shot)
                bgr = cv_from_pil(pil)
                pts = detect_points(bgr)
                log(f"等待检测: 当前彩点={len(pts)} (阈值 {MIN_POINTS_FOR_REAL_BOARD})")
                if len(pts) >= MIN_POINTS_FOR_REAL_BOARD:
                    return {"mode":"visual","screenshot":shot,"points":len(pts)}
            except Exception as e:
                log(f"截图或检测异常: {e}")
            time.sleep(1.2)
        # 超时：尝试解析捕获到的 responses
        log("视觉等待超时，尝试解析捕获到的网络响应")
        parsed = parse_network_responses_for_boards(responses)
        if parsed:
            return {"mode":"network","api_candidates":parsed}
        # 最后仍返回最后截图（视为未进入实盘）
        if last_shot:
            pil = pil_from_bytes(last_shot)
            bgr = cv_from_pil(pil)
            pts = detect_points(bgr)
            return {"mode":"visual","screenshot":last_shot,"points":len(pts)}
        return None
    finally:
        try:
            browser.close()
        except:
            pass

# ---------------- parsing network JSON heuristics ----------------
def find_lists_in_obj(obj):
    """递归找出候选的 list（长度>3 且元素为 dict 或 list）"""
    candidates=[]
    if isinstance(obj, list):
        if len(obj) >= 4:
            candidates.append(obj)
        for item in obj:
            candidates.extend(find_lists_in_obj(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            candidates.extend(find_lists_in_obj(v))
    return candidates

def parse_sequence_from_item(item):
    """
    给定一个 dict/list，尝试抽取 B/P 或 banker/player 或 0/1 序列
    返回 list of 'B'/'P' strings 或 None
    """
    seq = []
    if isinstance(item, list):
        for sub in item:
            res = parse_sequence_from_item(sub)
            if res:
                seq.extend(res if isinstance(res,list) else [res])
        if seq:
            return seq
    elif isinstance(item, dict):
        # common fields
        lower_keys = {k.lower():v for k,v in item.items()}
        # check winner-like fields
        for k in ("winner","result","outcome","type","side","hand","banker","player"):
            if k in lower_keys:
                v = lower_keys[k]
                if isinstance(v, str):
                    if v.lower().startswith("b") or "bank" in v.lower():
                        return ["B"]
                    if v.lower().startswith("p") or "player" in v.lower() or "闲" in v:
                        return ["P"]
                if isinstance(v, (int,float)):
                    # sometimes 1/2 mapping — guess: 1 banker, 2 player
                    if int(v) == 1: return ["B"]
                    if int(v) == 2: return ["P"]
        # if dict contains a list of moves
        for v in item.values():
            res = parse_sequence_from_item(v)
            if res:
                return res
    return None

def parse_network_responses_for_boards(responses):
    """
    responses: list of {"url":..., "status":..., "text":...}
    返回 parsed_boards: list of per-board sequences (each is list of 'B'/'P')
    """
    parsed_boards = []
    raw_candidates = []
    for r in responses:
        text = r.get("text","")
        if not text: continue
        # try json load
        try:
            j = json.loads(text)
        except Exception:
            # sometimes text includes JSON in HTML - skip
            continue
        lists = find_lists_in_obj(j)
        for lst in lists:
            # try parse each element into 'B'/'P' sequence
            seq = []
            for item in lst:
                res = parse_sequence_from_item(item)
                if res:
                    if isinstance(res,list): seq.extend(res)
                    else: seq.append(res)
            # if we get a lot of B/P markers, keep
            if len(seq) >= 8:
                parsed_boards.append(seq)
                raw_candidates.append({"url": r.get("url"), "sample": lst[:6]})
    # Post-process: if parsed_boards empty, try to find arrays of simple tokens
    if not parsed_boards:
        for r in responses:
            text = r.get("text","")
            if not text: continue
            # look for patterns like ["B","P","B",...]
            try:
                j = json.loads(text)
                if isinstance(j, list) and all(isinstance(x,str) for x in j) and len(j)>=8:
                    # normalize tokens
                    seq = []
                    for x in j:
                        s = x.strip().upper()
                        if s.startswith("B") or "BANK" in s.upper() or s=="庄": seq.append("B")
                        elif s.startswith("P") or "PLAY" in s.upper() or s=="闲": seq.append("P")
                    if len(seq)>=8:
                        parsed_boards.append(seq)
                        raw_candidates.append({"url": r.get("url"), "sample": j[:8]})
            except Exception:
                pass
    # return parsed_boards as list of sequences
    if parsed_boards:
        return {"parsed": parsed_boards, "raw_candidates": raw_candidates}
    return None

# ---------------- convert sequences into board_stats (compatible with visual) ----------------
def boards_from_sequences(seq_lists):
    """seq_lists: list of sequences of 'B'/'P' -> produce board_stats list"""
    boards = []
    for seq in seq_lists:
        # split into columns like visual flattening assumption: attempt 6 columns
        # heuristic: chunk into columns of length ~ ceil(len/6)
        n = len(seq)
        cols = 6
        col_h = math.ceil(n/cols)
        columns = []
        for c in range(cols):
            start = c*col_h; end = start+col_h
            col = seq[start:end]
            if col:
                columns.append(col)
        # flattened
        flattened = []
        maxlen = max((len(col) for col in columns), default=0)
        for r in range(maxlen):
            for col in columns:
                if r < len(col): flattened.append(col[r])
        # runs
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
        multiRuns = sum(1 for r in runs if r["len"]>=4)
        # detect consec3 in columns
        cols_max = []
        for col in columns:
            m=0
            if col:
                cur=col[0]; ln=1
                for s in col[1:]:
                    if s==cur: ln+=1
                    else:
                        m=max(m,ln); cur=s; ln=1
                m=max(m,ln)
            cols_max.append(m)
        consec3=False
        if len(cols_max) >= 3:
            for i in range(len(cols_max)-2):
                if cols_max[i]>=4 and cols_max[i+1]>=4 and cols_max[i+2]>=4:
                    consec3=True; break
        boards.append({"total":len(flattened),"maxRun":maxRun,"category":cat,"runs":runs,"multiRuns":multiRuns,"cols_max":cols_max,"consec3":consec3})
    return boards

# ---------------- main flow ----------------
def main():
    log("开始检测循环（最终版）")
    state = load_state()
    result=None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                result = try_visit_and_capture(p, url)
                if result:
                    break
            except Exception as e:
                log(f"访问 {url} 异常: {e}")
                continue
    summary = {"ts": nowstr(), "mode": None, "info": None, "boards":[]}
    overall=None; longCount=0; superCount=0
    # handle result
    if not result:
        log("未能通过任何方式获得数据")
        save_state(state)
        summary["mode"]="none"
        Path(LAST_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if result.get("mode") == "visual":
        pts = result.get("points", 0)
        summary["mode"]="visual"
        summary["points"]=pts
        log(f"视觉模式：截图彩点={pts}")
        # if we have screenshot and enough points do visual analysis
        if pts >= 8:
            pil = pil_from_bytes(result["screenshot"])
            bgr = cv_from_pil(pil)
            regions = cluster_regions(detect_points(bgr), bgr.shape[1], bgr.shape[0])
            # fallback: if cluster_regions returned empty, build single region covering entire
            if not regions:
                regions=[(0,0,bgr.shape[1], bgr.shape[0])]
            boards=[]
            for r in regions:
                st = analyze_region(bgr, r)
                boards.append(st)
            overall, longCount, superCount = classify_overall(boards)
            summary["boards"]=boards
        else:
            # too few points to be reliable; mark as failed visual
            summary["info"]="视觉截图点数过少"
            # try network parsing? (we already attempted in try_visit)
            summary["boards"]=[]
            overall="胜率中等（平台收割中等时段）"
    elif result.get("mode") == "network":
        summary["mode"]="network"
        parsed = result.get("api_candidates", {})
        summary["raw_candidates"] = parsed.get("raw_candidates", [])[:6]
        seqs = parsed.get("parsed", [])
        boards = boards_from_sequences(seqs)
        summary["boards"]=boards
        overall, longCount, superCount = classify_overall(boards)
        log(f"网络模式解析到 {len(boards)} 桌，判定 {overall}")
    else:
        summary["mode"]="unknown"
        summary["info"]="未知结果"
    # state transitions and Telegram
    now_iso = datetime.now(TZ).isoformat()
    was_active = state.get("active", False)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    if is_active_now and not was_active:
        # start
        history = state.get("history", [])
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        est = round(sum(durations)/len(durations)) if durations else 10
        est_end = (datetime.now(TZ) + timedelta(minutes=est)).strftime("%Y-%m-%d %H:%M")
        emoji="🚩"
        msg = f"{emoji} <b>DG 提醒 — {overall}</b>\n偵測時間 (MYT): {now_iso}\n長/超长龙桌數={longCount}，超长龙={superCount}\n估計結束時間: {est_end}（約 {est} 分鐘）"
        send_telegram(msg)
        state = {"active":True,"kind":overall,"start_time":now_iso,"last_seen":now_iso,"history":state.get("history", [])}
        save_state(state)
    elif is_active_now and was_active:
        state["last_seen"]=now_iso; state["kind"]=overall; save_state(state)
    elif (not is_active_now) and was_active:
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ); dur = round((end-start).total_seconds()/60)
        entry = {"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": dur}
        hist = state.get("history", []); hist.append(entry); hist = hist[-120:]
        new_state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":hist}
        save_state(new_state)
        emoji="✅"
        msg = f"{emoji} <b>DG 提醒 — {state.get('kind')} 已結束</b>\n開始: {entry['start_time']}\n結束: {entry['end_time']}\n實際持續: {dur} 分鐘"
        send_telegram(msg)
    else:
        save_state(state)
    # write summary
    summary["overall"]= overall
    summary["longCount"]= longCount
    summary["superCount"]= superCount
    summary["ts"]= nowstr()
    Path(LAST_SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("写入 last_run_summary.json")
    return

# small wrappers
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"主程式异常: {e}")
        raise
