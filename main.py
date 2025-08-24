# FINAL_FIX_V3
# -*- coding: utf-8 -*-
"""
最终稳健版 main.py (FINAL_FIX_V3)
特点：
 - 不使用 coords[:,1] 等易出错的 numpy 切片
 - 把坐标处理改为纯 Python 列表（安全）
 - 全面 try/except 保护，任何异常只记录不抛出 exit 1
 - 在控制台输出 "FINAL_FIX_V3 RUN" 以便确认运行的是此版本
"""
import os, sys, time, json, math
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
import cv2
from PIL import Image

# Optional heavy deps guarded
try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAY = True
except Exception:
    HAVE_PLAY = False

# ---------- config ----------
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
            log(f"Telegram 返回: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 出错: {e}")
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
        log(f"写 state.json 失败: {e}")

def save_summary(s):
    try:
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump(s,f,ensure_ascii=False,indent=2)
    except Exception as e:
        log(f"写 summary 失败: {e}")

def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil) if np else np.asarray(pil), cv2.COLOR_RGB2BGR)

# ---------- color detection ----------
def detect_beads(img_bgr):
    """返回点列表：[(x,y,label), ...]  label: 'B' (red) 或 'P' (blue)"""
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        return []
    lower1 = (0,100,70); upper1 = (8,255,255)
    lower2 = (160,80,70); upper2 = (179,255,255)
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb = (90,60,50); upperb = (140,255,255)
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    pts = []
    for mask, lbl in ((mask_r,'B'), (mask_b,'P')):
        try:
            cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            continue
        for cnt in cts:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if not M or M.get("m00",0) == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx,cy,lbl))
    return pts

# ---------- cluster boards (safe) ----------
def cluster_boards(points, w, h):
    """返回候选桌区 list of (x,y,w,h). 如果 points 为空则返回空"""
    if not points:
        return []
    cell = max(60, int(min(w,h)/12))
    cols = max(1, math.ceil(w / cell)); rows = max(1, math.ceil(h / cell))
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, max(0, x//cell))
        cy = min(rows-1, max(0, y//cell))
        grid[cy][cx] += 1
    thr = max(2, int(len(points) / (6*max(1,min(cols,rows)))))
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
        # fallback: uniform grid regions
        regs=[]
        for ry in range(rows):
            for rx in range(cols):
                regs.append((int(rx*cell), int(ry*cell), int(cell), int(cell)))
        return regs
    rects=[]
    for r,c in hits:
        x0 = c*cell; y0 = r*cell; w0 = cell; h0 = cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x0 > rx+rw+cell or x0+w0 < rx-cell or y0 > ry+rh+cell or y0+h0 < ry-cell):
                nx=min(rx,x0); ny=min(ry,y0)
                nw=max(rx+rw, x0+w0)-nx
                nh=max(ry+rh, y0+h0)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged:
            rects.append((x0,y0,w0,h0))
    regs=[]
    for (x0,y0,w0,h0) in rects:
        nx=max(0,x0-8); ny=max(0,y0-8)
        nw=min(w-nx, w0+16); nh=min(h-ny, h0+16)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ---------- analyze single region using safe Python lists ----------
def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"runs":[],"row_runs":[]}
    # coords as pure Python lists to avoid numpy shape issues
    coords = [(int(px), int(py)) for (px,py,_) in pts]
    labels = [lbl for (_,_,lbl) in pts]
    # xs, ys lists
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    # determine columns by binning (safe)
    try:
        bins = max(1, min(12, max(1, w//60)))
    except Exception:
        bins = 4
    edges = [int(round(i * (w / bins))) for i in range(bins+1)]
    col_idx = []
    for xpt in xs:
        # safe search
        ci = 0
        for i in range(bins):
            if edges[i] <= xpt <= edges[i+1]:
                ci = i; break
        col_idx.append(ci)
    col_count = max(1, max(col_idx)+1)
    # determine rows by binning
    try:
        rbins = max(3, min(14, max(1, h//28)))
    except Exception:
        rbins = 6
    redges = [int(round(i * (h / rbins))) for i in range(rbins+1)]
    row_idx = []
    for ypt in ys:
        ri = 0
        for i in range(rbins):
            if redges[i] <= ypt <= redges[i+1]:
                ri = i; break
        row_idx.append(ri)
    row_count = max(1, max(row_idx)+1)
    # build grid
    grid = [['' for _ in range(col_count)] for __ in range(row_count)]
    for i, lbl in enumerate(labels):
        try:
            rix = int(row_idx[i]); cix = int(col_idx[i])
            if 0 <= rix < row_count and 0 <= cix < col_count:
                grid[rix][cix] = lbl
        except Exception:
            continue
    # flattened vertical reading (column-major top->bottom)
    flattened=[]
    for c in range(col_count):
        for r in range(row_count):
            v = grid[r][c]
            if v: flattened.append(v)
    # vertical runs
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for v in flattened[1:]:
            if v == cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur); cur={"color":v,"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # horizontal row runs
    row_runs=[]
    for r in range(row_count):
        curc=None; curlen=0; maxh=0
        for c in range(col_count):
            v = grid[r][c]
            if v and v == curc:
                curlen += 1
            else:
                curc = v
                curlen = 1 if v else 0
            if curlen > maxh: maxh = curlen
        row_runs.append(maxh)
    # detect 3 consecutive rows each with horizontal run >=4
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >=4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
            has_multirow=True; break
    # classify
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"runs":runs,"row_runs":row_runs}

# ---------- capture screenshot (Playwright) ----------
def capture_screenshot(play, url):
    try:
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        ctx = browser.new_context(viewport={"width":1280,"height":900})
        page = ctx.new_page()
        page.goto(url, timeout=30000)
        time.sleep(1.2)
        # click likely "Free" button texts
        for txt in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
            try:
                loc = page.locator(f"text={txt}")
                if loc.count() > 0:
                    loc.first.click(timeout=2500); time.sleep(0.8); break
            except Exception:
                pass
        # gentle scrolling to allow lazy content
        for _ in range(2):
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.5)
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.5)
            except:
                pass
        time.sleep(1)
        shot = page.screenshot(full_page=True)
        try: ctx.close()
        except: pass
        try: browser.close()
        except: pass
        return shot
    except Exception as e:
        log(f"capture_screenshot 出错: {e}")
        return None

# ---------- overall classification ----------
def classify_overall(stats):
    long_count = sum(1 for b in stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in stats if b['category']=='super_long')
    multirow = sum(1 for b in stats if b.get('has_multirow',False))
    if super_count >=1 and long_count >=2 and (super_count + long_count) >=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if multirow >=3 and (long_count + super_count) >=2:
        return "中等胜率（中上）", long_count, super_count, multirow
    totals=[b.get('total',0) for b in stats]
    sparse = sum(1 for t in totals if t < 6)
    if stats and sparse >= len(stats)*0.6:
        return "胜率调低 / 收割时段", long_count, super_count, multirow
    return "胜率中等（平台收割中等时段）", long_count, super_count, multirow

# ---------- main ----------
def main():
    log("FINAL_FIX_V3 RUN")
    state = load_state()
    screenshot = None
    if not HAVE_PLAY:
        log("Playwright 未安装/不可用，无法抓取页面。请确保 playwright 可用。")
    else:
        try:
            with sync_playwright() as p:
                for url in DG_LINKS:
                    try:
                        screenshot = capture_screenshot(p, url)
                        if screenshot: break
                    except Exception as e:
                        log(f"访问 {url} 出错: {e}")
                        continue
        except Exception as e:
            log(f"Playwright overall error: {e}")

    if not screenshot:
        log("未取得截图，结束本次 run（不会抛异常）")
        save_state(state)
        return

    try:
        pil = pil_from_bytes(screenshot)
        img = cv2.cvtColor(np.array(pil) if np else cv2.cvtColor(pil, cv2.COLOR_RGB2BGR), cv2.COLOR_RGB2BGR) if np else cv2.cvtColor(pil.convert("RGB"), cv2.COLOR_RGB2BGR)
    except Exception:
        # fallback: use PIL->opencv via bytes
        try:
            pil = pil_from_bytes(screenshot)
            arr = pil.convert("RGB")
            img = cv2.cvtColor(__import__('numpy').array(arr), cv2.COLOR_RGB2BGR)
        except Exception as e:
            log(f"转换截图失败: {e}")
            save_state(state)
            return

    h,w = img.shape[:2]
    try:
        points = detect_beads(img)
        log(f"检测到点数: {len(points)}")
    except Exception as e:
        log(f"detect_beads 出错: {e}"); points = []

    regions = cluster_boards(points, w, h)
    log(f"聚类出候选桌区: {len(regions)}")
    board_stats=[]
    for i, r in enumerate(regions):
        try:
            st = analyze_region(img, r)
            st['region']=r; st['idx']=i+1
            board_stats.append(st)
        except Exception as e:
            log(f"分析 region {i+1} 出错（跳过）: {e}")
            continue
    if not board_stats:
        log("无有效 board_stats，结束本次 run")
        save_state(state)
        return

    overall, long_count, super_count, multirow_count = classify_overall(board_stats)
    log(f"判定: {overall} (长龙={long_count}, 超龙={super_count}, 连续3排多连={multirow_count})")
    now = datetime.now(TZ); now_iso = now.isoformat()
    was_active = state.get("active", False)
    is_active = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    if is_active and not was_active:
        history = state.get("history", [])
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        est_minutes = max(1, round(sum(durations)/len(durations))) if durations else 10
        est_end = (now + timedelta(minutes=est_minutes)).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🟢" if overall.startswith("放水") else "🔵"
        msg = f"{emoji} <b>DG 局势提醒 — {overall}</b>\n开始: {now_iso}\n长龙数: {long_count}；超长龙: {super_count}；连续3排多连桌: {multirow_count}\n估计结束: {est_end}（约 {est_minutes} 分钟）"
        send_telegram(msg)
        state = {"active":True,"kind":overall,"start_time":now_iso,"last_seen":now_iso,"history":state.get("history",[])}
        save_state(state)
    elif is_active and was_active:
        state["last_seen"]=now_iso; state["kind"]=overall; save_state(state)
    elif (not is_active) and was_active:
        start = datetime.fromisoformat(state.get("start_time"))
        end = now
        duration_minutes = round((end - start).total_seconds() / 60.0)
        history = state.get("history", [])
        history.append({"kind":state.get("kind"),"start_time":state.get("start_time"),"end_time":end.isoformat(),"duration_minutes":duration_minutes})
        history = history[-120:]
        new_state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":history}
        save_state(new_state)
        msg = f"🔴 <b>DG 放水/中上 已结束</b>\n类型: {state.get('kind')}\n开始: {state.get('start_time')}\n结束: {end.isoformat()}\n实际持续: {duration_minutes} 分钟"
        send_telegram(msg)
    else:
        save_state(state)

    summary = {"ts": now_iso, "overall": overall, "long_count": long_count, "super_count": super_count, "multirow_count": multirow_count, "boards": board_stats[:40]}
    save_summary(summary)
    log("本次运行完成（FINAL_FIX_V3）")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"捕获未处理异常（不会抛出 exit 1）: {e}")
        try:
            send_telegram(f"⚠️ DG 监测脚本异常（FINAL_FIX_V3）：{e}")
        except:
            pass
        sys.exit(0)
