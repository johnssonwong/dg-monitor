# -*- coding: utf-8 -*-
"""
DG 监测脚本（合规版 — 不破解滑块）
流程：
 - 访问 DG 链接并尝试点击 Free
 - 检测是否需要滑块/安全条（若需要则截图并发到 Telegram 提醒手动完成）
 - 若进入实盘（通过图像检测判断），按规则分析桌面并在满足“放水”或“中等胜率(中上)”时发 Telegram 提醒
 - 每次运行为一次检测（在 GitHub Actions 中可设置每5分钟运行）
注意：此脚本不会尝试自动破解滑块或验证码。
"""

import os, sys, time, json, traceback, random
from io import BytesIO
from datetime import datetime, timedelta, timezone
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import numpy as np
except Exception:
    np = None

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAY = True
except Exception:
    HAVE_PLAY = False

# config
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

# image params
RED_RANGES = [((0,100,70),(8,255,255)), ((160,80,70),(179,255,255))]
BLUE_RANGE = ((90,60,50),(140,255,255))
MIN_CONTOUR_AREA = 8
CELL_MIN = 60
ROW_BIN_H = 28
POINTS_THRESH_FOR_REAL_TABLE = 12  # 若检测到珠点 >= 12，视为进入实盘（可调）

def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

def send_tg_msg(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，无法发送消息")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML"}, timeout=20)
        return r.ok
    except Exception as e:
        log("send msg fail: " + str(e))
        return False

def send_tg_photo(bytes_img, caption=""):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，无法发送图片")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        files = {"photo": ("shot.jpg", bytes_img)}
        data = {"chat_id":TG_CHAT, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log("send photo fail: " + str(e))
        return False

def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def pil_to_bytes(pil):
    bio = BytesIO(); pil.save(bio, format="JPEG", quality=85); bio.seek(0); return bio.read()

def cv_from_pil(pil):
    arr = np.array(pil) if np else None
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# detect beads
def detect_beads(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r = None
    for lo,hi in RED_RANGES:
        p = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_r = p if mask_r is None else (mask_r | p)
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

# simple clustering to find boxes
def cluster_boards(points, w, h):
    if not points:
        return []
    cell = max(CELL_MIN, int(min(w,h)/12))
    cols = max(1, (w+cell-1)//cell)
    rows = max(1, (h+cell-1)//cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, max(0, x//cell))
        cy = min(rows-1, max(0, y//cell))
        grid[cy][cx]+=1
    thr = max(2, int(len(points)/(6*max(1,min(cols,rows)))))
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
        regs=[]
        for ry in range(rows):
            for rx in range(cols):
                regs.append((int(rx*cell), int(ry*cell), int(cell), int(cell)))
        return regs
    rects=[]
    for r,c in hits:
        x0 = c*cell; y0=r*cell; w0=cell; h0=cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x0 > rx+rw+cell or x0+w0 < rx-cell or y0 > ry+rh+cell or y0+h0 < ry-cell):
                nx=min(rx,x0); ny=min(ry,y0)
                nw=max(rx+rw, x0+w0)-nx; nh=max(ry+rh, y0+h0)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged:
            rects.append((x0,y0,w0,h0))
    regs=[]
    for x0,y0,w0,h0 in rects:
        nx=max(0,x0-12); ny=max(0,y0-12); nw=min(w-nx, w0+24); nh=min(h-ny, h0+24)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"row_runs":[],"runs":[]}
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; labels=[p[2] for p in pts]
    bins = max(1, min(12, int(max(1,w/60))))
    col_idx=[int(min(bins-1, max(0, int((xv / w) * bins)))) if w>0 else 0 for xv in xs]
    col_count = max(1, max(col_idx)+1)
    rbins = max(3, min(14, int(max(1,h/ROW_BIN_H))))
    row_idx=[int(min(rbins-1, max(0, int((yv / h) * rbins)))) if h>0 else 0 for yv in ys]
    row_count = max(1, max(row_idx)+1)
    grid=[['' for _ in range(col_count)] for __ in range(row_count)]
    for i,lbl in enumerate(labels):
        try:
            rix=int(row_idx[i]); cix=int(col_idx[i])
            if 0<=rix<row_count and 0<=cix<col_count:
                grid[rix][cix]=lbl
        except:
            continue
    flattened=[]
    for c in range(col_count):
        for r in range(row_count):
            v = grid[r][c]
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
    row_runs=[]
    for r in range(row_count):
        curc=None; curlen=0; maxh=0
        for c in range(col_count):
            v = grid[r][c]
            if v and v==curc:
                curlen+=1
            else:
                curc=v; curlen=1 if v else 0
            if curlen>maxh: maxh=curlen
        row_runs.append(maxh)
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i]>=4 and row_runs[i+1]>=4 and row_runs[i+2]>=4:
            has_multirow=True; break
    cat="other"
    if maxRun>=10: cat="super_long"
    elif maxRun>=8: cat="long"
    elif maxRun>=4: cat="longish"
    elif maxRun==1: cat="single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"row_runs":row_runs,"runs":runs}

def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow = sum(1 for b in board_stats if b.get('has_multirow',False))
    if super_count>=1 and long_count>=2 and (super_count+long_count)>=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if (long_count+super_count)>= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if multirow>=3 and (long_count+super_count)>=2:
        return "中等胜率（中上）", long_count, super_count, multirow
    totals=[b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if board_stats and sparse >= len(board_stats)*0.6:
        return "收割时段（胜率调低）", long_count, super_count, multirow
    return "胜率中等", long_count, super_count, multirow

def annotate_pil(pil, regions, stats):
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.load_default()
    except:
        f = None
    for i,r in enumerate(regions):
        x,y,w,h = r
        d.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        s = stats[i] if i < len(stats) else {}
        txt = f"#{i+1} {s.get('category','?')} run={s.get('maxRun',0)} multi={s.get('has_multirow',False)}"
        d.text((x+4,y+4), txt, fill=(255,255,0), font=f)
    return pil

# Playwright navigation (without attempting to solve slider)
def capture_page():
    if not HAVE_PLAY:
        log("Playwright 不可用")
        return None, None, "no_play"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
            ctx = browser.new_context(viewport={"width":1366,"height":900})
            page = ctx.new_page()
            last_err=None
            for url in DG_LINKS:
                try:
                    log("打开 " + url)
                    page.goto(url, timeout=30000)
                    time.sleep(1.0)
                    # try click free variants
                    for t in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
                        try:
                            sel = page.locator(f"text={t}")
                            if sel.count() > 0:
                                log("点击: " + t)
                                sel.first.click(timeout=4000)
                                time.sleep(0.8)
                                break
                        except Exception:
                            continue
                    # take screenshot after clicking
                    shot = page.screenshot(full_page=True)
                    pil = pil_from_bytes(shot); img_bgr = cv_from_pil(pil)
                    # quick detection: count beads
                    pts = detect_beads(img_bgr)
                    log("当前珠点数: %d" % len(pts))
                    # detect presence of slider-like elements (common selectors)
                    slider_found = False
                    try:
                        # check some common slider containers in page and frames
                        sel_names = ["#slider", ".geetest_slider", ".geetest_slider_button", ".vaptcha", ".captcha", ".nc_"]
                        for s in sel_names:
                            try:
                                if page.locator(s).count() > 0:
                                    slider_found = True; break
                            except:
                                pass
                        # also check for iframes containing 'geetest' or similar
                        for fr in page.frames:
                            try:
                                if "geetest" in fr.url or "captcha" in fr.url or fr.locator(".geetest_slider_button").count()>0:
                                    slider_found = True; break
                            except:
                                pass
                    except Exception:
                        pass
                    ctx.close(); browser.close()
                    status = "ok" if len(pts) >= POINTS_THRESH_FOR_REAL_TABLE else ("need_slider" if slider_found or len(pts) < POINTS_THRESH_FOR_REAL_TABLE else "not_entered")
                    return pil, img_bgr, status
                except Exception as e:
                    last_err = str(e); log("访问异常: " + str(e))
                    continue
            try:
                ctx.close()
            except: pass
            try:
                browser.close()
            except: pass
            return None, None, "all_fail:"+str(last_err)
    except Exception as e:
        log("Playwright outer exception: " + str(e))
        return None, None, "playouter:"+str(e)

def main_once():
    try:
        pil, img_bgr, status = capture_page()
        if pil is None:
            send_tg_msg(f"⚠️ DG 抓图失败，状态: {status}")
            log("抓图未成功，结束本次运行")
            return
        pts = detect_beads(img_bgr)
        log("最终珠点: %d" % len(pts))
        # if need manual slider completion:
        if status == "need_slider" or len(pts) < POINTS_THRESH_FOR_REAL_TABLE:
            # send screenshot + instructions to Telegram to ask user to manually complete slider
            send_tg_photo(pil_to_bytes(pil), caption=f"⚠️ 需要手动完成安全条/滑块才能进入实盘。请在手机/浏览器打开如下链接并完成滑块：\n{DG_LINKS[0]}\n（完成后，GitHub Actions 下次运行会继续检测。）\n检测时间: {nowstr()}")
            send_tg_msg("请手动打开上面链接并完成滑块/安全条。完成后脚本会在下一次检测时自动识别进入实盘并继续分析。")
            return
        # 已进入实盘，做聚类分析
        h,w = img_bgr.shape[:2]
        regions = cluster_boards(pts, w, h)
        board_stats=[]
        for r in regions:
            try:
                st = analyze_region(img_bgr, r)
            except Exception as e:
                st = {"total":0,"maxRun":0,"category":"error","has_multirow":False,"row_runs":[],"runs":[]}
            board_stats.append(st)
        overall, long_c, super_c, multi_c = classify_overall(board_stats)
        summary = {"ts": datetime.now(TZ).isoformat(), "status": status, "overall": overall, "long_count": long_c, "super_count": super_c, "multirow_count": multi_c}
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # annotate and send final screenshot
        pil_ann = annotate_pil(pil, regions, board_stats)
        send_tg_photo(pil_to_bytes(pil_ann), caption=f"DG 判定: {overall}\n长龙:{long_c} 超龙:{super_c} 连珠:{multi_c}\n时间:{nowstr()}")
        # only alert for the two desired states
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            emoji = "🟢" if overall.startswith("放水") else "🔵"
            send_tg_msg(f"{emoji} <b>{overall}</b>\n时间: {nowstr()}\n长龙:{long_c} 超龙:{super_c} 连珠桌:{multi_c}")
        log("检测完成: " + overall)
    except Exception as e:
        log("主流程异常: " + str(e))
        log(traceback.format_exc())
        try:
            send_tg_msg(f"⚠️ DG 监测脚本异常: {e}")
        except:
            pass

if __name__ == "__main__":
    main_once()
    # ensure zero exit
    sys.exit(0)
