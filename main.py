# -*- coding: utf-8 -*-
"""
DG 实盘监测器（用于 GitHub Actions 或本地）
实现已把中文函数名改为合法英文名 detect_multi_row。
"""

import os, sys, time, json, math
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright
from sklearn.cluster import KMeans

# --- 配置（可通过环境变量覆盖） ---
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()

DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))
DEFAULT_ESTIMATE_MINUTES = int(os.environ.get("DEFAULT_ESTIMATE_MINUTES", "10"))

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"

TZ = timezone(timedelta(hours=8))

def log(msg):
    print(f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def send_tg(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置，跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": text}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 提醒发送成功。")
            return True
        else:
            log(f"Telegram 返回错误: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(p):
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 110, 70]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([160,110,70]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb = np.array([90, 70, 60]); upperb = np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)

    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)

    pts = []
    for mask, label in [(mask_r,'B'), (mask_b,'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            pts.append((cx, cy, label))
    return pts

def cluster_into_regions(points, w, h):
    if len(points) == 0:
        return []
    cell = max(60, int(min(w,h)/12))
    cols = math.ceil(w/cell); rows = math.ceil(h/cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    thr = 6
    hits = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr:
                hits.append((r,c))
    rects=[]
    if hits:
        for (r,c) in hits:
            x0 = c*cell; y0 = r*cell; w0 = cell; h0 = cell
            placed=False
            for i,(rx,ry,rw,rh) in enumerate(rects):
                if not (x0 > rx+rw+cell or x0+w0 < rx-cell or y0 > ry+rh+cell or y0+h0 < ry-cell):
                    nx = min(rx, x0); ny = min(ry, y0)
                    nw = max(rx+rw, x0+w0) - nx
                    nh = max(ry+rh, y0+h0) - ny
                    rects[i] = (nx,ny,nw,nh)
                    placed=True
                    break
            if not placed:
                rects.append((x0,y0,w0,h0))
        regs=[(max(0,int(x-8)), max(0,int(y-8)), min(w,int(w0+16)), min(h,int(h0+16))) for (x,y,w0,h0) in rects]
        return regs
    pts_arr = np.array([[p[0], p[1]] for p in points])
    k = min(8, max(1, len(points)//8))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pts_arr)
    rects2=[]
    for lab in range(k):
        mpts = pts_arr[kmeans.labels_==lab]
        if len(mpts)==0: continue
        x0,y0 = mpts.min(axis=0); x1,y1 = mpts.max(axis=0)
        rects2.append((int(x0)-12,int(y0)-8, min(w,int(x1-x0+24)), min(h,int(y1-y0+16))))
    return rects2

def analyze_region(img, rect):
    x,y,w,h = rect
    crop = img[y:y+h, x:x+w]
    pts = detect_red_blue_points(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","columns":[],"flattened":[]}
    pts_sorted = sorted(pts, key=lambda z: z[0])
    xs = [p[0] for p in pts_sorted]
    col_groups=[]
    for i,p in enumerate(pts_sorted):
        px = p[0]; placed=False
        for g in col_groups:
            meanx = sum([pts_sorted[idx][0] for idx in g])/len(g)
            if abs(meanx - px) <= max(8, w//40):
                g.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    columns=[]
    for g in col_groups:
        col_pts = sorted([pts_sorted[idx] for idx in g], key=lambda z: z[1])
        seq = [p[2] for p in col_pts]
        columns.append(seq)
    maxlen = max((len(c) for c in columns), default=0)
    flattened=[]
    for r in range(maxlen):
        for c in columns:
            if r < len(c):
                flattened.append(c[r])
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for i in range(1,len(flattened)):
            if flattened[i]==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":flattened[i],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    perColMax=[]
    for col in columns:
        if not col: perColMax.append(0); continue
        m = 1; curc = col[0]; curlen=1
        for j in range(1,len(col)):
            if col[j]==curc: curlen+=1
            else:
                m = max(m, curlen); curc = col[j]; curlen = 1
        m = max(m, curlen)
        perColMax.append(m)
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    else: cat = "other"
    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "columns":perColMax, "flattened":flattened, "runs":runs}

def detect_multi_row(perColMax):
    """
    判断是否存在 '连续 3 列'，每列垂直最长同色 run >=4
    """
    if not perColMax: return False
    seq = perColMax
    count = 0
    for v in seq:
        if v >= 4:
            count += 1
            if count >= 3:
                return True
        else:
            count = 0
    return False

def classify_all(board_stats):
    longCount = sum(1 for b in board_stats if b["category"] in ("long","super_long"))
    superCount = sum(1 for b in board_stats if b["category"]=="super_long")
    longishCount = sum(1 for b in board_stats if b["category"]=="longish")
    multi_count = sum(1 for b in board_stats if detect_multi_row(b.get("columns", [])))
    n = len(board_stats)
    sparse = sum(1 for b in board_stats if b["total"] < 6)
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount, multi_count
    if multi_count >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount, multi_count
    if n>0 and sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount, multi_count
    return "胜率中等（平台收割中等时段）", longCount, superCount, multi_count

def capture_screenshot(play, url):
    browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
    try:
        context = browser.new_context(viewport={"width":1280, "height":900})
        page = context.new_page()
        log(f"打开 {url}")
        page.goto(url, timeout=35000)
        time.sleep(2)
        for txt in ["Free", "免费试玩", "免费", "免费玩", "Play Free", "试玩"]:
            try:
                el = page.locator(f"text={txt}")
                if el.count() > 0:
                    el.first.click(timeout=3500)
                    log(f"尝试点击: {txt}")
                    break
            except Exception:
                pass
        time.sleep(2)
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            time.sleep(0.5)
            page.mouse.wheel(0, 400)
            time.sleep(0.5)
            page.mouse.wheel(0, -400)
            time.sleep(1)
        except Exception:
            pass
        time.sleep(4)
        shot = page.screenshot(full_page=True)
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

def main():
    state = load_state()
    log("开始检测循环。")
    screenshot = None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                screenshot = capture_screenshot(p, url)
                if screenshot:
                    break
            except Exception as e:
                log(f"访问失败: {e}")
                continue
    if not screenshot:
        log("无法取得截图，本次结束。")
        save_state(state); return
    pil = pil_from_bytes(screenshot)
    img = cv_from_pil(pil)
    h,w = img.shape[:2]
    pts = detect_red_blue_points(img)
    log(f"检测到彩点: {len(pts)}")
    if len(pts) == 0:
        log("未检测到任何点（页面可能未加载或布局不同）。")
        save_state(state); return
    regions = cluster_into_regions(pts, w, h)
    log(f"聚类出候选桌: {len(regions)}")
    board_stats=[]
    for r in regions:
        st = analyze_region(img, r)
        board_stats.append(st)
    overall, longCount, superCount, multi_count = classify_all(board_stats)
    log(f"判定 => {overall} (长龙数={longCount}, 超长龙={superCount}, 连珠桌数={multi_count})")
    summary = {"ts": datetime.now(TZ).isoformat(), "overall": overall, "longCount": longCount, "superCount": superCount,
               "multi_count": multi_count, "boards": board_stats}
    with open(LAST_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    was_active = state.get("active", False)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    now_iso = datetime.now(TZ).isoformat()
    if is_active_now and not was_active:
        history = state.get("history", [])
        est_minutes = DEFAULT_ESTIMATE_MINUTES
        if history:
            durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
            if durations:
                est_minutes = round(sum(durations)/len(durations))
        est_end = (datetime.now(TZ) + timedelta(minutes=est_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        emoji = "📣" if overall=="放水时段（提高胜率）" else "🔔"
        text = (f"{emoji} [DG提醒] {overall} 開始\n時間(本地): {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"长龙数: {longCount}  超长龙: {superCount}  连珠桌数: {multi_count}\n"
                f"估計結束時間: {est_end}（約 {est_minutes} 分鐘）\n")
        send_tg(text)
        state = {"active": True, "kind": overall, "start_time": now_iso, "last_seen": now_iso, "history": history}
        save_state(state)
        log("事件開始，已发送提醒及保存狀態。")
    elif is_active_now and was_active:
        state["last_seen"] = now_iso
        state["kind"] = overall
        save_state(state)
        log("仍在活動中，已更新 last_seen。")
    elif (not is_active_now) and was_active:
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ)
        duration_min = round((end - start).total_seconds() / 60)
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_min})
        history = history[-200:]
        text = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end.isoformat()}\n實際持續: {duration_min} 分鐘"
        send_tg(text)
        state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history}
        save_state(state)
        log("事件已結束，發送結束通知並保存歷史。")
    else:
        save_state(state)
        log("目前不在放水/中上時段，不發提醒。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"脚本异常: {e}")
        raise
