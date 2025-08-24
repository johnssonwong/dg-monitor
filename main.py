# -*- coding: utf-8 -*-
"""
最终修复版 main.py — 强健防护，避免 IndexError
注意：请完整替换仓库中的 main.py 并 commit，然后手动 Run workflow 测试。
"""

import os, sys, time, json, math, random
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
import cv2
from PIL import Image

# Use lightweight clustering fallback; sklearn optional but not required for core safety
try:
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

from playwright.sync_api import sync_playwright

# ---------------- config ----------------
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("Telegram 未配置，跳过发送。")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML"}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log(f"Telegram API 返回: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"读取 state.json 失败: {e}")
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    try:
        with open(STATE_FILE,"w",encoding="utf-8") as f:
            json.dump(s,f,ensure_ascii=False,indent=2)
    except Exception as e:
        log(f"写入 state.json 失败: {e}")

def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ----------- color detection (robust) -------------
def detect_beads(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # two red ranges
    lower1 = np.array([0,100,70]); upper1=np.array([8,255,255])
    lower2 = np.array([160,80,70]); upper2=np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb = np.array([90,60,50]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    pts = []
    for mask, lbl in [(mask_r, 'B'), (mask_b, 'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            M = cv2.moments(cnt)
            if M.get("m00",0) == 0:
                continue
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            pts.append((cx, cy, lbl))
    return pts

# ---------- board clustering (safe) ----------
def cluster_boards(points, w, h):
    if not points:
        return []
    cell = max(60, int(min(w,h)/12))
    cols = math.ceil(w / cell); rows = math.ceil(h / cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    thr = max(3, int(len(points) / (6*max(1,min(cols,rows)))))
    hits=[]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr:
                hits.append((r,c))
    if not hits:
        # fallback divide into grid cells as regions
        regs = []
        for ry in range(rows):
            for rx in range(cols):
                regs.append((rx*cell, ry*cell, cell, cell))
        return regs
    rects=[]
    for (r,c) in hits:
        x = c*cell; y = r*cell; wcell = cell; hcell = cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+wcell < rx-cell or y > ry+rh+cell or y+hcell < ry-cell):
                nx = min(rx,x); ny = min(ry,y)
                nw = max(rx+rw, x+wcell) - nx
                nh = max(ry+rh, y+hcell) - ny
                rects[i] = (nx,ny,nw,nh)
                merged=True
                break
        if not merged:
            rects.append((x,y,wcell,hcell))
    regs=[]
    for (x,y,w0,h0) in rects:
        nx=max(0,x-10); ny=max(0,y-10)
        nw=min(w-nx, w0+20); nh=min(h-ny, h0+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ---------- analyze a single board safely ----------
def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"runs":[],"row_runs":[]}
    # coords with forced shape
    coords = np.array([[p[0], p[1]] for p in pts], dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1,2)
    elif coords.size == 0:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"runs":[],"row_runs":[]}
    labels = [p[2] for p in pts]
    # determine columns: try KMeans if available and enough points
    col_idx = None
    col_count = 1
    try:
        if _HAVE_SK and len(coords) >= 8:
            k = min(12, max(2, len(coords)//4))
            km = KMeans(n_clusters=k, random_state=0).fit(coords[:,0].reshape(-1,1))
            centroids = km.cluster_centers_.flatten()
            order = sorted(range(len(centroids)), key=lambda i: centroids[i])
            map_order = {orig: i for i,orig in enumerate(order)}
            col_idx = [map_order[int(lab)] if isinstance(lab,(np.integer,int)) else int(lab) for lab in km.labels_]
            col_count = len(order)
        else:
            raise Exception("skip kmeans cols")
    except Exception:
        # fallback simple binning by x
        bins = max(1, min(8, int(max(1,w) / 60)))
        edges = np.linspace(0, max(1,w), bins+1)
        xs = coords[:,0]
        col_idx = np.clip(np.searchsorted(edges, xs) - 1, 0, bins-1).tolist()
        col_count = bins

    # determine rows: try kmeans on y
    row_idx = None
    row_count = 1
    try:
        if _HAVE_SK and len(coords) >= 8:
            r = min(12, max(3, len(coords)//4))
            ky = KMeans(n_clusters=r, random_state=0).fit(coords[:,1].reshape(-1,1))
            centers = sorted([c[0] for c in ky.cluster_centers_])
            row_idx = [int(np.argmin([abs(coords[i,1]-c) for c in centers])) for i in range(len(coords))]
            row_count = len(centers)
        else:
            raise Exception("skip kmeans rows")
    except Exception:
        bins = max(3, min(12, int(max(1,h) / 28)))
        edges = np.linspace(0, max(1,h), bins+1)
        ys = coords[:,1]
        row_idx = np.clip(np.searchsorted(edges, ys) - 1, 0, bins-1).tolist()
        row_count = bins

    # build grid row_count x col_count
    grid = [['' for _ in range(col_count)] for __ in range(row_count)]
    for i, lbl in enumerate(labels):
        try:
            rx = int(row_idx[i]); cx = int(col_idx[i])
            if 0 <= rx < row_count and 0 <= cx < col_count:
                grid[rx][cx] = lbl
        except Exception:
            continue

    # flattened vertical reading (column-major top->bottom)
    flattened = []
    for c in range(col_count):
        for r in range(row_count):
            v = grid[r][c]
            if v:
                flattened.append(v)

    # compute vertical runs
    runs = []
    if flattened:
        cur = {"color": flattened[0], "len": 1}
        for v in flattened[1:]:
            if v == cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur)
                cur = {"color": v, "len": 1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)

    # compute horizontal row runs
    row_runs = []
    for r in range(row_count):
        curc = None; curlen = 0; maxh = 0
        for c in range(col_count):
            cc = grid[r][c]
            if cc and cc == curc:
                curlen += 1
            else:
                curc = cc
                curlen = 1 if cc else 0
            if curlen > maxh:
                maxh = curlen
        row_runs.append(maxh)
    # detect 3 consecutive rows each with horizontal run >=4
    has_multirow = False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >= 4 and row_runs[i+1] >= 4 and row_runs[i+2] >= 4:
            has_multirow = True
            break

    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"

    return {"total": len(flattened), "maxRun": maxRun, "category": cat, "has_multirow": has_multirow, "runs": runs, "row_runs": row_runs}

# ---------- screenshot with Playwright ----------
def capture_screenshot(play, url):
    try:
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        ctx = browser.new_context(viewport={"width":1280,"height":900})
        page = ctx.new_page()
        page.goto(url, timeout=30000)
        time.sleep(2)
        # try click common Free buttons
        for txt in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
            try:
                el = page.locator(f"text={txt}")
                if el.count() > 0:
                    el.first.click(timeout=3000)
                    time.sleep(1)
                    break
            except Exception:
                pass
        # scroll a bit
        for _ in range(3):
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.6)
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.6)
            except:
                pass
        time.sleep(2)
        shot = page.screenshot(full_page=True)
        try: ctx.close()
        except: pass
        try: browser.close()
        except: pass
        return shot
    except Exception as e:
        log(f"capture_screenshot 失败: {e}")
        try:
            browser.close()
        except:
            pass
        return None

# ---------- overall classification ----------
def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow_count = sum(1 for b in board_stats if b.get('has_multirow',False))
    # 超长龙触发型
    if super_count >= 1 and long_count >= 2 and (super_count + long_count) >= 3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    if multirow_count >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if board_stats and sparse >= len(board_stats)*0.6:
        return "胜率调低 / 收割时段", long_count, super_count, multirow_count
    return "胜率中等（平台收割中等时段）", long_count, super_count, multirow_count

# ---------------- main ----------------
def main():
    state = load_state()
    log("开始一次检测")
    screenshot = None
    try:
        with sync_playwright() as p:
            for url in DG_LINKS:
                try:
                    screenshot = capture_screenshot(p, url)
                    if screenshot:
                        break
                except Exception as e:
                    log(f"访问 {url} 失败: {e}")
                    continue
    except Exception as e:
        log(f"Playwright 启动失败: {e}")
        save_state(state)
        return

    if not screenshot:
        log("未取得截图，结束本次run")
        save_state(state)
        return

    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    try:
        points = detect_beads(bgr)
    except Exception as e:
        log(f"detect_beads 失败: {e}")
        points = []
    log(f"检测到点数: {len(points)}")
    regions = cluster_boards(points, w, h)
    log(f"聚类出 {len(regions)} 个候选桌区")
    board_stats = []
    for i, r in enumerate(regions):
        try:
            st = analyze_region(bgr, r)
            st['region'] = r
            st['idx'] = i+1
            board_stats.append(st)
        except Exception as e:
            log(f"分析 region {i+1} 失败（已忽略）：{e}")
            continue
    if not board_stats:
        log("没有可用 board_stats，结束")
        save_state(state)
        return
    overall, long_count, super_count, multirow_count = classify_overall(board_stats)
    log(f"本次判定：{overall} (长龙={long_count} 超龙={super_count} 连续3排多连={multirow_count})")
    now = datetime.now(TZ); now_iso = now.isoformat()
    was_active = state.get("active", False)
    is_active = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    if is_active and not was_active:
        history = state.get("history", [])
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        est_minutes = max(1, round(sum(durations)/len(durations))) if durations else 10
        est_end = (now + timedelta(minutes=est_minutes)).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🟢" if overall.startswith("放水") else "🔵"
        msg = f"{emoji} <b>DG 局势提醒 — {overall}</b>\n开始: {now_iso}\n长龙数: {long_count}；超长龙: {super_count}；连续3排多连桌数: {multirow_count}\n估计结束: {est_end}（约 {est_minutes} 分钟）"
        send_telegram(msg)
        state = {"active":True,"kind":overall,"start_time":now_iso,"last_seen":now_iso,"history":state.get("history",[])}
        save_state(state)
    elif is_active and was_active:
        state["last_seen"] = now_iso; state["kind"] = overall; save_state(state)
    elif (not is_active) and was_active:
        start = datetime.fromisoformat(state.get("start_time"))
        end = now
        duration_minutes = round((end - start).total_seconds()/60.0)
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_minutes})
        history = history[-120:]
        new_state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":history}
        save_state(new_state)
        msg = f"🔴 <b>DG 放水/中上 已结束</b>\n类型: {state.get('kind')}\n开始: {state.get('start_time')}\n结束: {end.isoformat()}\n实际持续: {duration_minutes} 分钟"
        send_telegram(msg)
    else:
        save_state(state)
    # write summary for debugging
    summary = {"ts": now_iso, "overall": overall, "long_count": long_count, "super_count": super_count, "multirow_count": multirow_count, "boards": board_stats[:40]}
    try:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"写summary失败: {e}")
    log("本次运行完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"捕获未处理异常: {e}")
        try:
            send_telegram(f"⚠️ DG 监测脚本异常：{e}")
        except:
            pass
        # 不 raise，避免 Action 以 exit code 1 结束
        sys.exit(0)
