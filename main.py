# -*- coding: utf-8 -*-
"""
DG 监测脚本（改进版）
- 修复 env int 解析错误
- 强化滑块尝试（随机化、重试、UA/JS 指纹弱化）
- 当实时抓取失败或未进入实盘（点数过少）时，尝试以历史事件（state.json / historical_events.json）作为替补触发提醒
- 精确记录开始/结束时间并发送 Telegram（开始含估算结束时间；结束含真实持续分钟）
"""
import os, sys, time, json, math, random, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright

# ---------- 配置与环境解析 ----------
def int_env(name, default):
    v = os.environ.get(name, "")
    try:
        return int(v) if v is not None and v != "" else int(default)
    except:
        return int(default)

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "").strip()

DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

MIN_BOARDS_FOR_PAW = int_env("MIN_BOARDS_FOR_PAW", 3)
MID_LONG_REQ = int_env("MID_LONG_REQ", 2)
MID_MULTI_ROW_REQ = int_env("MID_MULTI_ROW_REQ", 3)
COOLDOWN_MINUTES = int_env("COOLDOWN_MINUTES", 10)

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"
HISTORICAL_FILE = "historical_events.json"

# Malaysia timezone
TZ = timezone(timedelta(hours=8))

def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

# ---------- Telegram ----------
def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置，跳过发送。")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log(f"Telegram API 返回: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 出错: {e}")
        return False

# ---------- State management ----------
def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json_file(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_state():
    return load_json_file(STATE_FILE, {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []})

def save_state(s):
    save_json_file(STATE_FILE, s)

# ---------- Image helpers ----------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,100,90]); upper1 = np.array([10,255,255])
    lower2 = np.array([160,100,90]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb = np.array([95, 70, 50]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    points=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            M = cv2.moments(c)
            if M["m00"]==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

def mean(a): return sum(a)/len(a) if a else 0

# ---------- Clustering boards & analyzing ----------
def cluster_boards(points, img_w, img_h):
    if not points: return []
    cell = max(60, int(min(img_w,img_h)/12))
    cols = math.ceil(img_w/cell); rows = math.ceil(img_h/cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,c) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    hits=[]
    thr = max(3, int(cell/30))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr: hits.append((r,c))
    if not hits:
        # fallback: make a few generic regions
        regs=[]
        wstep = img_w//4; hstep = img_h//3
        for i in range(4):
            for j in range(3):
                regs.append((i*wstep+10, j*hstep+10, wstep-20, hstep-20))
        return regs
    rects=[]
    for r,c in hits:
        x=c*cell; y=r*cell; w=cell; h=cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                nx=min(rx,x); ny=min(ry,y)
                nw=max(rx+rw, x+w)-nx; nh=max(ry+rh, y+h)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged: rects.append((x,y,w,h))
    regs=[]
    for (x,y,w,h) in rects:
        nx=max(0,x-8); ny=max(0,y-8); nw=min(img_w-nx,w+16); nh=min(img_h-ny,h+16)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_board_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts,_,_ = detect_red_blue_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "flattened":[], "runs":[], "multi_row":False}
    xs = [p[0] for p in pts]
    ids = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for i in ids:
        xv = xs[i]; placed=False
        for grp in col_groups:
            gv = [pts[j][0] for j in grp]; ifv = mean(gv)
            if abs(ifv - xv) <= max(8, w//40):
                grp.append(i); placed=True; break
        if not placed: col_groups.append([i])
    sequences=[]
    for grp in col_groups:
        col_pts = sorted([pts[i] for i in grp], key=lambda t: t[1])
        seq = [t[2] for t in col_pts]; sequences.append(seq)
    flattened=[]
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col): flattened.append(col[r])
    runs=[]
    if flattened:
        cur={"color":flattened[0], "len":1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur["color"]: cur["len"]+=1
            else: runs.append(cur); cur={"color":flattened[k],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    if maxRun >= 10: cat="super_long"
    elif maxRun >= 8: cat="long"
    elif maxRun >= 4: cat="longish"
    elif maxRun == 1: cat="single"
    else: cat="other"
    # detect multi_row: at least 3 consecutive columns with top run >=4
    multi_row=False
    try:
        col_run_lengths=[]
        for seq in sequences:
            top_run=1
            for i in range(1,len(seq)):
                if seq[i]==seq[i-1]: top_run+=1
                else: break
            col_run_lengths.append(top_run)
        cons=0
        for rl in col_run_lengths:
            if rl>=4:
                cons+=1
                if cons>=3:
                    multi_row=True; break
            else:
                cons=0
    except:
        multi_row=False
    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "flattened":flattened, "runs":runs, "multi_row":multi_row}

# ---------- Slider solving helpers ----------
def human_like_move(page, start, end, steps=30):
    sx, sy = start; ex, ey = end
    for i in range(1, steps+1):
        t = i/steps
        x = sx + (ex-sx)*(t**0.9) + random.uniform(-3,3)
        y = sy + (ey-sy)*(t**0.9) + random.uniform(-2,2)
        try:
            page.mouse.move(x,y)
        except: pass
        time.sleep(random.uniform(0.007,0.018))

def try_solve_slider(page):
    try:
        # try several selector candidates
        cands = ["div[role=slider]", ".slider .handler", ".slide-block", ".vaptcha_slider", ".geetest_slider_button","#slider",".drag-handle",".drag"]
        for sel in cands:
            try:
                els = page.query_selector_all(sel)
                if els and len(els)>0:
                    el = els[0]
                    box = el.bounding_box()
                    if not box: continue
                    sx = box["x"] + box["width"]*0.2
                    sy = box["y"] + box["height"]/2
                    ex = box["x"] + box["width"]*0.95
                    ey = sy
                    page.mouse.move(sx,sy); page.mouse.down()
                    human_like_move(page, (sx,sy),(ex,ey), steps=random.randint(20,45))
                    page.mouse.up()
                    time.sleep(random.uniform(1.0,2.0))
                    return True
            except Exception:
                continue
        # fallback: generic drag across middle of viewport
        vp = page.viewport_size or {"width":1280,"height":800}
        sx = vp["width"]*0.16; sy = vp["height"]*0.6
        ex = vp["width"]*0.84; ey = sy
        page.mouse.move(sx,sy); page.mouse.down()
        human_like_move(page,(sx,sy),(ex,ey), steps=random.randint(30,60))
        page.mouse.up()
        time.sleep(random.uniform(1.0,1.8))
        return True
    except Exception as e:
        log(f"try_solve_slider exception: {e}")
        return False

# ---------- Capture DG screenshot (attempts slider) ----------
def capture_dg_screenshot(play, url, attempts=2):
    browser = None
    try:
        # reduce headless fingerprinting
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width":1280,"height":800},
                                      user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36")
        page = context.new_page()
        page.set_extra_http_headers({"Accept-Language":"zh-CN,zh;q=0.9,en;q=0.8"})
        page.goto(url, timeout=35000)
        time.sleep(1.2 + random.random()*1.6)
        # click Free / 免费试玩
        clicked=False
        txts = ["Free","免费试玩","免费","Play Free","试玩","free","Go In"]
        for t in txts:
            try:
                els = page.get_by_text(t)
                if els.count()>0:
                    els.first.click(timeout=3000); clicked=True; log(f"clicked text {t}"); break
            except: pass
        if not clicked:
            # try common buttons
            for sel in ["button", "a"]:
                try:
                    elements = page.query_selector_all(sel)
                    for el in elements:
                        try:
                            txt = (el.inner_text() or "").strip()
                            if txt and any(k in txt for k in ["Free","免费","Play","试玩"]):
                                el.click(); clicked=True; log(f"clicked element with text '{txt}'"); break
                        except: continue
                    if clicked: break
                except: continue
        time.sleep(1.0 + random.random()*1.7)
        # try slider multiple times
        success=False
        for i in range(4):
            ok = try_solve_slider(page)
            log(f"slider attempt {i+1} -> {ok}")
            time.sleep(1.0 + random.random()*1.4)
            # quick check: take a small screenshot and see if colored points appear
            try:
                shot = page.screenshot()
                pil = pil_from_bytes(shot); bgr = cv_from_pil(pil)
                pts,_,_ = detect_red_blue_points(bgr)
                if len(pts) > 8:
                    log("见到较多彩点（认为已进入实盘界面）")
                    success=True
                    break
            except Exception:
                pass
        # final: if success True => return full-page shot; otherwise return last shot for debug (or None)
        full = page.screenshot(full_page=True)
        return full
    except Exception as e:
        log(f"capture_dg_screenshot exception: {e}")
        return None
    finally:
        try:
            if browser: browser.close()
        except: pass

# ---------- classify & fallback logic ----------
def classify_boards(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    multi_row_count = sum(1 for b in board_stats if b.get('multi_row', False))
    n = len(board_stats)
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount, multi_row_count
    if multi_row_count >= MID_MULTI_ROW_REQ and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount, multi_row_count
    if n>0 and sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount, multi_row_count
    return "胜率中等（平台收割中等时段）", longCount, superCount, multi_row_count

def fallback_using_history(now_dt, state):
    """
    若实时抓取失败或点数过少，则用 state.history 和 historical_events.json
    计算当前时间是否落入历史上常见“放水时间窗口”：
    - 使用最近 4 周（state.history & historical_events.json）事件统计
    - 计算每分钟-of-week (0..10079) 出现事件次数，若当前 minute 有 >=阈值比例(例如 >=30% days) 则视为 fallback 放水
    返回 (is_fallback_active, kind, est_minutes)
    """
    # collect events from state.history and historical file
    events = []
    hist_state = state.get("history", []) or []
    for h in hist_state:
        try:
            st = datetime.fromisoformat(h.get("start_time"))
            dur = int(h.get("duration_minutes",0))
            events.append({"start": st, "duration": dur})
        except:
            continue
    hist_file = load_json_file(HISTORICAL_FILE, [])
    for h in hist_file:
        try:
            st = datetime.fromisoformat(h.get("start_time"))
            dur = int(h.get("duration_minutes",0))
            events.append({"start": st, "duration": dur})
        except:
            continue
    # only keep last 4 weeks events
    cutoff = now_dt - timedelta(weeks=4)
    events = [e for e in events if e["start"] >= cutoff]
    if len(events) < 3:
        return False, None, None
    # build minute-of-week histogram
    counts = {}
    for e in events:
        minute = (e["start"].weekday()*24*60) + (e["start"].hour*60 + e["start"].minute)
        counts[minute] = counts.get(minute, 0) + 1
    # smooth: consider +/- 10 minutes neighborhood
    aggregated = {}
    for m,v in counts.items():
        for d in range(-10,11):
            mm = (m + d) % (7*24*60)
            aggregated[mm] = aggregated.get(mm,0) + v
    cur_minute = (now_dt.weekday()*24*60) + (now_dt.hour*60 + now_dt.minute)
    # threshold: if aggregated[cur_minute] is in top X percentile or >= some absolute count
    maxv = max(aggregated.values()) if aggregated else 0
    val = aggregated.get(cur_minute, 0)
    # ratio vs max
    if maxv>0 and (val >= max(2, int(0.35*maxv))):
        # estimate avg duration among events near this minute
        nearby = []
        for e in events:
            diff = abs(((e["start"].weekday()*24*60 + e["start"].hour*60 + e["start"].minute) - cur_minute))
            if abs(diff) <= 15: nearby.append(e["duration"])
        if not nearby: est = 10
        else: est = int(round(sum(nearby)/len(nearby)))
        # decide kind: if many long events -> 放水
        avgdur = est
        if avgdur >= 8:
            kind = "放水时段（提高胜率）"
        else:
            kind = "中等胜率（中上）"
        return True, kind, est
    return False, None, None

# ---------- 主流程 ----------
def main():
    log("开始一次检测循环（改进版）")
    state = load_state()
    now = datetime.now(TZ)
    screenshot = None
    # 1) 尝试实时抓取
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                shot = capture_dg_screenshot(p, url)
                if shot:
                    screenshot = shot
                    break
            except Exception as e:
                log(f"访问 {url} 失败: {e}")
    # 若有截图则尝试识别
    if screenshot:
        pil = pil_from_bytes(screenshot)
        bgr = cv_from_pil(pil)
        h,w = bgr.shape[:2]
        pts,_,_ = detect_red_blue_points(bgr)
        log(f"检测到彩点数: {len(pts)}")
        if len(pts) >= 8:
            regions = cluster_boards(pts, w, h)
            log(f"聚类候选桌数: {len(regions)}")
            board_stats=[]
            for r in regions:
                st = analyze_board_region(bgr, r)
                board_stats.append(st)
            overall, longCount, superCount, multi_row_count = classify_boards(board_stats)
            log(f"实时判定结果: {overall} (长龙数={longCount}, 超={superCount}, multi_row={multi_row_count})")
            # 保存 summary
            debug = {"ts": now.isoformat(), "overall": overall, "longCount": longCount, "superCount": superCount, "multi_row": multi_row_count, "boards": board_stats[:40]}
            save_json_file(LAST_SUMMARY, debug)
            # 状态变迁
            was_active = state.get("active", False)
            is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
            if is_active_now and not was_active:
                # 开始事件：估算结束时间（基于历史）
                hist = state.get("history", [])
                est_minutes = None
                if hist:
                    durations = [h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
                    if durations: est_minutes = int(round(sum(durations)/len(durations)))
                if not est_minutes: est_minutes = 10
                est_end = (now + timedelta(minutes=est_minutes)).strftime("%Y-%m-%d %H:%M:%S")
                emoji = "🚨"
                msg = f"{emoji} <b>DG提醒 — {overall}</b>\n偵測時間: {now.strftime('%Y-%m-%d %H:%M:%S')}\n長/超龍桌數: {longCount}/{superCount}\n多排多連桌數: {multi_row_count}\n估計結束時間: {est_end}（約 {est_minutes} 分鐘）"
                send_telegram(msg)
                state = {"active": True, "kind": overall, "start_time": now.isoformat(), "last_seen": now.isoformat(), "history": state.get("history", [])}
                save_state(state)
            elif is_active_now and state.get("active", False):
                state["last_seen"] = now.isoformat(); state["kind"] = overall; save_state(state)
            elif (not is_active_now) and state.get("active", False):
                # 事件结束
                try:
                    start = datetime.fromisoformat(state.get("start_time"))
                    duration = (now - start).total_seconds()/60.0
                    duration_min = int(round(duration))
                except:
                    duration_min = 0
                history = state.get("history", [])
                history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": now.isoformat(), "duration_minutes": duration_min})
                history = history[-200:]
                new_state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history}
                save_state(new_state)
                emoji = "🔔"
                msg = f"{emoji} <b>DG提醒 — {state.get('kind')} 已結束</b>\n開始: {state.get('start_time')}\n結束: {now.isoformat()}\n實際持續: {duration_min} 分鐘"
                send_telegram(msg)
            else:
                save_state(state)
            return
        else:
            log("截图检测到点数过少（可能未进入实盘或界面变化），进入 fallback 检查。")
            save_json_file(LAST_SUMMARY, {"ts": now.isoformat(), "note":"few_points", "points": len(pts)})
    else:
        log("无法取得实时截图，进入 fallback 检查。")

    # ---------- fallback: 使用最近历史 / historical_events.json ----------
    fb_ok, fb_kind, fb_est = fallback_using_history(now, state)
    if fb_ok:
        log(f"历史替补判定：{fb_kind}（估计 {fb_est} 分钟）")
        # 若当前状态非 active（或不同种类），发提醒并记录开始
        if not state.get("active", False):
            est_end = (now + timedelta(minutes=fb_est)).strftime("%Y-%m-%d %H:%M:%S")
            emoji = "⚠️"
            msg = f"{emoji} <b>DG 替補提醒 — {fb_kind}</b>\n（注：由歷史數據替補判定，因無法即時進入實盤）\n偵測時間: {now.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束時間: {est_end}（約 {fb_est} 分鐘）"
            ok = send_telegram(msg)
            new_state = {"active": True, "kind": fb_kind, "start_time": now.isoformat(), "last_seen": now.isoformat(), "history": state.get("history", [])}
            save_state(new_state)
        else:
            log("已有活動中，不重複替補提醒。")
    else:
        log("替補歷史未命中（歷史事件不足或時間不匹配），不提醒。")
        # 如果处于活动中但实时/历史都未命中，保持 state（不强制结束）
        save_state(state)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"未處理例外: {e}\n{traceback.format_exc()}")
        sys.exit(1)
