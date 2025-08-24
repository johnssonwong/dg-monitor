# -*- coding: utf-8 -*-
"""
修复版 main.py — 更稳健的 DG 实盘检测 (GitHub Actions)
修复点：
 - 防止 coords 变成一维（对 coords 强制 reshape(-1,2)）
 - 对 KMeans / 数组索引添加回退与 try/except
 - 每个 region 分析出错仅记录，不中断整个 run
 - 全程捕获未处理异常，避免 exit code 1
 - 更稳健的“连续3排多连”检测逻辑
"""
import os, sys, time, json, math, random
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from playwright.sync_api import sync_playwright

# env / config
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW","3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))
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
        log(f"发送 Telegram 失败: {e}")
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s,f,ensure_ascii=False,indent=2)

def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# 更稳健的色点检测
def detect_beads(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,100,70]); upper1=np.array([8,255,255])
    lower2 = np.array([160,80,70]); upper2=np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb = np.array([90,60,50]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    k=np.ones((3,3),np.uint8)
    mask_r=cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b=cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    pts=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area<8: continue
            M=cv2.moments(cnt)
            if M["m00"]==0: continue
            cx=int(M["m10"]/M["m00"]); cy=int(M["m01"]/M["m00"])
            pts.append((cx,cy,label))
    return pts

def cluster_boards(points, w, h):
    if not points:
        return []
    cell = max(60, int(min(w,h)/12))
    cols = math.ceil(w/cell); rows=math.ceil(h/cell)
    grid=[[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx=min(cols-1, x//cell); cy=min(rows-1, y//cell)
        grid[cy][cx]+=1
    thr=max(3, int(len(points)/(6*max(1,min(cols,rows)))))
    hits=[]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]>=thr:
                hits.append((r,c))
    if not hits:
        # fallback simple partition
        regions=[]
        for ry in range(rows):
            for cx in range(cols):
                regions.append((cx*cell, ry*cell, cell, cell))
        return regions
    rects=[]
    for (r,c) in hits:
        x=c*cell; y=r*cell; wcell=cell; hcell=cell
        placed=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+wcell < rx-cell or y > ry+rh+cell or y+hcell < ry-cell):
                nx=min(rx,x); ny=min(ry,y)
                nw=max(rx+rw, x+wcell)-nx
                nh=max(ry+rh, y+hcell)-ny
                rects[i]=(nx,ny,nw,nh); placed=True; break
        if not placed:
            rects.append((x,y,wcell,hcell))
    regs=[]
    for (x,y,w0,h0) in rects:
        nx=max(0,x-10); ny=max(0,y-10)
        nw=min(w-nx, w0+20); nh=min(h-ny, h0+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"runs":[]}
    coords = np.array([[p[0], p[1]] for p in pts], dtype=float).reshape(-1,2)  # 强制reshape
    labels = [p[2] for p in pts]
    # columns: 尝试 KMeans on x；若失败 fallback to binning
    try:
        X = coords[:,0].reshape(-1,1)
        k = min(max(2, int(len(coords)/3)), 12)
        if len(coords) < 6:
            raise Exception("too few points for KMeans cols")
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        col_order = sorted(range(k), key=lambda i: km.cluster_centers_[i][0])
        col_of_point = [int(np.where(km.labels_[i]==km.labels_[i])[0]) if False else int(km.labels_[i]) for i in range(len(X))]
        # remap labels to order
        order_map = {orig: idx for idx,orig in enumerate(col_order)}
        col_idx_for_point = [order_map.get(ci, 0) for ci in col_of_point]
    except Exception:
        # fallback bin by x
        bins = max(1, min(8, int(w/60)))
        xs = coords[:,0]
        edges = np.linspace(0, w, bins+1)
        col_idx_for_point = np.clip(np.searchsorted(edges, xs) - 1, 0, bins-1).tolist()
        k = bins
    # rows: cluster y into row centers
    try:
        Ys = coords[:,1].reshape(-1,1)
        rcount = min(max(3, int(h/28)), max(3, len(coords)))
        if len(coords) < 6:
            raise Exception("too few for KMeans rows")
        ky = KMeans(n_clusters=rcount, random_state=0).fit(Ys)
        row_centers = sorted([c[0] for c in ky.cluster_centers_])
        row_idx_for_point = [int(np.argmin([abs(p[1]-rc) for rc in row_centers])) for p in coords]
        row_count = len(row_centers)
    except Exception:
        # fallback bin by y
        bins = max(3, min(10, int(h/30)))
        ys = coords[:,1]
        edges = np.linspace(0, h, bins+1)
        row_idx_for_point = np.clip(np.searchsorted(edges, ys) - 1, 0, bins-1).tolist()
        row_count = bins
    col_count = max(1, max(col_idx_for_point)+1)
    # build grid
    grid = [['' for _ in range(col_count)] for __ in range(row_count)]
    for i,(cx,cy,lbl) in enumerate(pts):
        try:
            rix = int(row_idx_for_point[i])
            cix = int(col_idx_for_point[i])
            if 0 <= rix < row_count and 0 <= cix < col_count:
                grid[rix][cix] = lbl
        except Exception:
            continue
    # compute flattened vertical reading and runs (column-major top->bottom)
    flattened=[]
    for col in range(col_count):
        # collect items in this column sorted by row index
        for row in range(row_count):
            v = grid[row][col]
            if v: flattened.append(v)
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for v in flattened[1:]:
            if v==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":v,"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # horizontal row runs -> 找到每行最大横向同色连
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
    # check 3 consecutive rows each with horizontal run >=4
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >=4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
            has_multirow=True
            break
    # classification
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"runs":runs,"row_runs":row_runs}

def capture_screenshot(play, url):
    try:
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        context = browser.new_context(viewport={"width":1280,"height":900})
        page = context.new_page()
        page.goto(url, timeout=30000)
        time.sleep(2)
        # click likely buttons
        texts = ["Free","免费试玩","免费","Play Free","试玩","进入"]
        for t in texts:
            try:
                el = page.locator(f"text={t}")
                if el.count()>0:
                    el.first.click(timeout=3000); time.sleep(1); break
            except Exception:
                continue
        # try scrolls
        for _ in range(3):
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.6)
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.6)
                page.mouse.wheel(0,400)
                time.sleep(0.4)
            except:
                pass
        time.sleep(2)
        shot = page.screenshot(full_page=True)
        try: context.close()
        except: pass
        try: browser.close()
        except: pass
        return shot
    except Exception as e:
        log(f"截图失败: {e}")
        try:
            browser.close()
        except:
            pass
        return None

def classify_overall(stats):
    long_count = sum(1 for b in stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in stats if b['category']=='super_long')
    multirow_count = sum(1 for b in stats if b.get('has_multirow',False))
    # 超长龙触发型
    if super_count >=1 and long_count >=2 and (super_count+long_count) >=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高勝率）", long_count, super_count, multirow_count
    if multirow_count >=3 and (long_count + super_count) >=2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    totals=[b['total'] for b in stats]
    sparse=sum(1 for t in totals if t < 6)
    if stats and sparse >= len(stats)*0.6:
        return "胜率调低 / 收割时段", long_count, super_count, multirow_count
    return "胜率中等（平台收割中等时段）", long_count, super_count, multirow_count

def main():
    state = load_state()
    log("开始检测循环")
    screenshot=None
    try:
        with sync_playwright() as p:
            for url in DG_LINKS:
                try:
                    screenshot = capture_screenshot(p, url)
                    if screenshot: break
                except Exception as e:
                    log(f"访问 {url} 失败: {e}")
                    continue
    except Exception as e:
        log(f"Playwright 启动失败: {e}")
        save_state(state); return

    if not screenshot:
        log("未取得截图，结束此次 run")
        save_state(state); return

    pil=pil_from_bytes(screenshot); bgr=cv_from_pil(pil)
    h,w=bgr.shape[:2]
    try:
        points = detect_beads(bgr)
        log(f"检测到点数: {len(points)}")
    except Exception as e:
        log(f"检测点失败: {e}"); points=[]
    regions = cluster_boards(points, w, h)
    log(f"聚类出候选桌区: {len(regions)}")
    board_stats=[]
    for idx,r in enumerate(regions):
        try:
            st = analyze_region(bgr, r)
            st['region'] = r
            st['idx'] = idx+1
            board_stats.append(st)
        except Exception as e:
            log(f"分析第{idx+1}区出错，但继续: {e}")
            continue
    if not board_stats:
        log("未提取到board stats，结束")
        save_state(state); return
    overall, long_count, super_count, multirow_count = classify_overall(board_stats)
    log(f"判断: {overall} (長={long_count} 超={super_count} multirow={multirow_count})")
    now = datetime.now(TZ); now_iso = now.isoformat()
    was_active = state.get("active", False)
    is_active = overall in ("放水时段（提高胜率）","中等胜率（中上）")
    if is_active and not was_active:
        # start event
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
    # save summary
    summary = {"ts": now_iso, "overall": overall, "long_count": long_count, "super_count": super_count, "multirow_count": multirow_count, "boards": board_stats[:40]}
    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("本次运行完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 捕获得到任何未预见异常，记录但不抛出，确保 Action 不以 1 退出
        log(f"捕获到未处理异常（已记录，但不抛出以避免 exit code 1）: {e}")
        try:
            # 尝试发 Telegram 报错提示（可选）
            send_telegram(f"⚠️ DG 监测脚本遇到异常：{str(e)}")
        except:
            pass
        # 结束但不 raise
        sys.exit(0)
