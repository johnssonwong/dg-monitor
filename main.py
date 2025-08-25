# dg_monitor_final.py
# -*- coding: utf-8 -*-
"""
DG 实盘检测脚本 — 最终版（发送截图到 Telegram，判定严格按用户规则）
把本文件保存为 main.py 并在 GitHub Actions 或服务器上运行。
环境变量（必须/可选）:
 - TG_BOT_TOKEN (必须)
 - TG_CHAT_ID   (必须)
 - MIN_BOARDS_FOR_PAW (默认 3)
 - MID_LONG_REQ (默认 2)
"""

import os, sys, time, math, json, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

# optional numpy & sklearn; script works with/without sklearn fallback
try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

# playwright import
try:
    from playwright.sync_api import sync_playwright
    _HAVE_PLAY = True
except Exception:
    _HAVE_PLAY = False

# --------------------
# CONFIG (可调)
# --------------------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

# Color detection HSV thresholds (可按场景微调)
RED_RANGES = [((0, 100, 70), (8, 255, 255)), ((160, 80, 70), (179, 255, 255))]
BLUE_RANGE = ((90, 60, 50), (140, 255, 255))
MIN_CONTOUR_AREA = 8   # 小点过滤阈值，若环境噪点多，可增大到 12-20
CELL_MIN = 60          # 用于聚类网格尺寸的最小单元（可按分辨率调）
ROW_BIN_H = 28         # 水平方向（y）bin size
# --------------------

def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def send_telegram_message(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，跳过 send message")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML"}, timeout=20)
        return r.ok
    except Exception as e:
        log(f"send message fail: {e}")
        return False

def send_telegram_photo(bytes_img, caption=""):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，跳过 send photo")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        files = {"photo": ("screenshot.jpg", bytes_img)}
        data = {"chat_id": TG_CHAT, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log(f"send photo fail: {e}")
        return False

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,"r",encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s,f,ensure_ascii=False,indent=2)

# -------- image helpers ----------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def pil_to_bytes(pil, fmt="JPEG"):
    bio = BytesIO()
    pil.save(bio, fmt, quality=85)
    bio.seek(0)
    return bio.read()

def cv_from_pil(pil):
    arr = np.array(pil) if np else None
    if arr is None:
        # fallback via bytes
        b = pil.tobytes()
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ---------- detection primitives ----------
def detect_beads_opencv(img_bgr):
    """返回 list of (x,y,label) label: 'B'=red('庄'), 'P'=blue('闲')"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r = None
    for lo, hi in RED_RANGES:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_r = part if mask_r is None else (mask_r | part)
    mask_b = cv2.inRange(hsv, np.array(BLUE_RANGE[0]), np.array(BLUE_RANGE[1]))
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    pts=[]
    for mask, lbl in [(mask_r,'B'), (mask_b,'P')]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA: continue
            M = cv2.moments(c)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx,cy,lbl))
    return pts

# ---------- cluster boards ----------
def cluster_boards_safe(points, w, h):
    """简单将桌子聚成若干 region，返回 list of (x,y,w,h)"""
    if not points:
        return []
    cell = max(CELL_MIN, int(min(w,h)/12))
    cols = max(1, math.ceil(w / cell)); rows = max(1, math.ceil(h / cell))
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, max(0, x//cell))
        cy = min(rows-1, max(0, y//cell))
        grid[cy][cx]+=1
    thr = max(2, int(len(points) / (6*max(1,min(cols,rows)))))
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
        # fallback - return uniform grid boxes
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
                nw=max(rx+rw, x0+w0)-nx; nh=max(ry+rh, y0+h0)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged:
            rects.append((x0,y0,w0,h0))
    # expand a little
    regs=[]
    for x0,y0,w0,h0 in rects:
        nx=max(0,x0-10); ny=max(0,y0-10); nw=min(w-nx, w0+20); nh=min(h-ny, h0+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ---------- analyze a region ----------
def analyze_region_strict(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads_opencv(crop)  # pts local coords
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"row_runs":[],"runs":[]}
    # convert to lists
    coords = [(p[0], p[1]) for p in pts]
    labels = [p[2] for p in pts]
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    # column bin by x
    bins = max(1, min(12, int(max(1,w/60))))
    edges = [i*(w/bins) for i in range(bins+1)]
    col_idx = []
    for xv in xs:
        ci = int(min(bins-1, max(0, int((xv / w) * bins))))
        col_idx.append(ci)
    col_count = max(1, max(col_idx)+1)
    # rows bins by y
    rbins = max(3, min(14, int(max(1,h/ROW_BIN_H))))
    redges = [i*(h/rbins) for i in range(rbins+1)]
    row_idx=[]
    for yv in ys:
        ri = int(min(rbins-1, max(0, int((yv / h) * rbins))))
        row_idx.append(ri)
    row_count = max(1, max(row_idx)+1)
    # build grid row_count x col_count
    grid = [['' for _ in range(col_count)] for __ in range(row_count)]
    for i,lbl in enumerate(labels):
        try:
            rix = int(row_idx[i]); cix = int(col_idx[i])
            if 0 <= rix < row_count and 0 <= cix < col_count:
                grid[rix][cix] = lbl
        except:
            continue
    # vertical flattened reading (column-major top->bottom)
    flattened=[]
    for c in range(col_count):
        for r in range(row_count):
            if grid[r][c]:
                flattened.append(grid[r][c])
    # vertical runs
    runs=[]
    if flattened:
        cur = {"color":flattened[0],"len":1}
        for v in flattened[1:]:
            if v==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":v,"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # horizontal row_runs
    row_runs=[]
    for r in range(row_count):
        curc=None; curlen=0; maxh=0
        for c in range(col_count):
            v = grid[r][c]
            if v and v==curc:
                curlen+=1
            else:
                curc=v
                curlen = 1 if v else 0
            if curlen > maxh: maxh = curlen
        row_runs.append(maxh)
    # multirow 连珠: 连续 3 排每排横向连 >=4
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >=4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
            has_multirow=True; break
    # classification for this board
    cat="other"
    if maxRun >= 10: cat="super_long"
    elif maxRun >= 8: cat="long"
    elif maxRun >= 4: cat="longish"
    elif maxRun == 1: cat="single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"row_runs":row_runs,"runs":runs}

# ---------- overall classification (严格按你定义) ----------
def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow_count = sum(1 for b in board_stats if b.get('has_multirow',False))
    # 规则 1: 若有 1 个超长龙 + 至少 2 个长龙 且 总数 >=3 -> 放水
    if super_count >=1 and long_count >=2 and (super_count + long_count) >=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    # 规则 2: 若总共长龙/超长龙 >= MIN_BOARDS_FOR_PAW -> 放水（桌数混合）
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    # 规则 3: 连珠/多连触发中等胜率（中上）
    if multirow_count >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    # 规则 4: 如果桌面大多数空 (point总数少) -> 收割时段
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if board_stats and sparse >= len(board_stats)*0.6:
        return "收割时段（胜率调低）", long_count, super_count, multirow_count
    # 其他 -> 胜率中等
    return "胜率中等", long_count, super_count, multirow_count

# ---------- annotate image ----------
def annotate_and_pack(pil_img, regions, board_stats):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    for i, r in enumerate(regions):
        x,y,w,h = r
        draw.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        st = board_stats[i]
        txt = f"#{i+1} {st['category']} run={st['maxRun']} multi={st['has_multirow']}"
        draw.text((x+4, y+4), txt, fill=(255,255,0), font=font)
    return pil_img

# ---------- capture function ----------
def capture_dg_screenshot():
    if not _HAVE_PLAY:
        log("Playwright 未安装，不可抓取页面。")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
            ctx = browser.new_context(viewport={"width":1280,"height":900})
            page = ctx.new_page()
            for url in DG_LINKS:
                try:
                    page.goto(url, timeout=30000)
                    time.sleep(1)
                    # try clicking Free buttons
                    for t in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
                        try:
                            el = page.locator(f"text={t}")
                            if el.count() > 0:
                                el.first.click(timeout=2500); time.sleep(0.8); break
                        except:
                            pass
                    time.sleep(1.0)
                    shot = page.screenshot(full_page=True)
                    try: ctx.close()
                    except: pass
                    try: browser.close()
                    except: pass
                    return shot
                except Exception as e:
                    log(f"访问 {url} 失败: {e}")
                    continue
    except Exception as e:
        log(f"Playwright 全局错误: {e}")
    return None

# ---------- main ----------
def main_once():
    state = load_state()
    screenshot = capture_dg_screenshot()
    if not screenshot:
        log("未抓到截图，本次结束")
        return
    pil = Image.open(BytesIO(screenshot)).convert("RGB")
    # convert to opencv
    arr = np.array(pil); img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h,w = img.shape[:2]
    # detect beads globally first
    points_all = detect_beads_opencv(img)
    log(f"检测到总点数: {len(points_all)}")
    # cluster possible board regions
    regions = cluster_boards_safe(points_all, w, h)
    log(f"聚类出 {len(regions)} 候选桌子")
    board_stats=[]
    for idx, r in enumerate(regions):
        try:
            st = analyze_region_strict(img, r)
        except Exception as e:
            st = {"total":0,"maxRun":0,"category":"error","has_multirow":False,"row_runs":[],"runs":[]}
        board_stats.append(st)
    # classify
    overall, long_count, super_count, multirow_count = classify_overall(board_stats)
    now = datetime.now(TZ).isoformat()
    summary = {"ts": now, "overall": overall, "long_count": long_count, "super_count": super_count, "multirow_count": multirow_count, "boards": board_stats[:40]}
    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # annotate screenshot and send to telegram if needed
    pil_annot = pil.copy()
    pil_annot = annotate_and_pack(pil_annot, regions, board_stats)
    bytes_img = BytesIO(); pil_annot.save(bytes_img, format="JPEG", quality=85); bytes_img.seek(0)
    caption = f"DG 监测: {overall}\n长龙:{long_count} 超龙:{super_count} 连珠桌:{multirow_count}\n时间:{now}"
    # send always a summary shot to Telegram so you can check
    ok = send_telegram_photo(bytes_img.read(), caption=caption)
    if not ok:
        log("发送 Telegram 图片失败")
    else:
        log("已发送截图到 Telegram")
    # if overall indicates 放水或 中等胜率(中上) -> send a textual alert too (highlight)
    if overall in ("放水时段（提高胜率）", "中等胜率（中上）"):
        emoji = "🟢" if overall.startswith("放水") else "🔵"
        msg = f"{emoji} <b>{overall}</b>\n开始: {now}\n长龙数:{long_count} 超龙:{super_count} 连珠桌:{multirow_count}"
        send_telegram_message(msg)
    # save state transitions
    was_active = state.get("active", False)
    is_active = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    if is_active and not was_active:
        state = {"active":True,"kind":overall,"start_time":now,"last_seen":now,"history":state.get("history",[])}
        save_state(state)
    elif is_active and was_active:
        state["last_seen"] = now; state["kind"] = overall; save_state(state)
    elif (not is_active) and was_active:
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.fromisoformat(now)
        duration_minutes = round((end - start).total_seconds()/60.0)
        history = state.get("history", [])
        history.append({"kind":state.get("kind"),"start_time":state.get("start_time"),"end_time":now,"duration_minutes":duration_minutes})
        state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":history}
        save_state(state)
        send_telegram_message(f"🔴 放水/中上 已结束: {state.get('kind')} 持续 {duration_minutes} 分钟")
    else:
        save_state(state)
    log(f"本次检测完成 -> {overall}")

if __name__ == "__main__":
    try:
        main_once()
    except Exception as e:
        log("主流程异常:" + str(e))
        log(traceback.format_exc())
        try:
            send_telegram_message(f"⚠️ DG 监测脚本异常: {e}")
        except:
            pass
        sys.exit(0)
