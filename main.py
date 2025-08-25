# FINAL_SLIDER_V1
# -*- coding: utf-8 -*-
"""
DG 监测 - 带滑块/安全条自动处理（FINAL_SLIDER_V1）
功能概述：
 - 点击 Free / 免费试玩
 - 在主 frame + 所有 iframe 中查找滑块/拖动控件并尝试模拟拖动（多策略）
 - 如果检测到已进入实盘桌面（通过图像珠点检测 & 聚类判断），则截图并发 Telegram 警报
 - 避免 numpy 切片 IndexError，全面防护
注意：替换仓库的 main.py 后手动 Run workflow 测试一次
"""

import os, sys, time, math, json, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import numpy as np
except Exception:
    np = None

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PLAY = True
except Exception:
    HAVE_PLAY = False

# config
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

# image detection params (可在需要时微调)
RED_RANGES = [((0,100,70),(8,255,255)), ((160,80,70),(179,255,255))]
BLUE_RANGE = ((90,60,50),(140,255,255))
MIN_CONTOUR_AREA = 8
CELL_MIN = 60
ROW_BIN_H = 28

def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def send_tg_msg(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，跳过 send msg")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML"}, timeout=20)
        return r.ok
    except Exception as e:
        log(f"send msg fail: {e}")
        return False

def send_tg_photo(bytes_img, caption=""):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，跳过 send photo")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        files = {"photo": ("shot.jpg", bytes_img)}
        data = {"chat_id":TG_CHAT, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log(f"send photo fail: {e}")
        return False

def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    if np:
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        arr = pil.tobytes()
        return cv2.imdecode(np.frombuffer(arr, np.uint8), cv2.IMREAD_COLOR)

# ---------- simple bead detection ----------
def detect_beads(img_bgr):
    """返回 list of (x,y,label)"""
    hsl = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r = None
    for lo, hi in RED_RANGES:
        part = cv2.inRange(hsl, np.array(lo), np.array(hi))
        mask_r = part if mask_r is None else (mask_r | part)
    mask_b = cv2.inRange(hsl, np.array(BLUE_RANGE[0]), np.array(BLUE_RANGE[1]))
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

# ---------- clustering to regions ----------
def cluster_boards(points, w, h):
    if not points:
        return []
    cell = max(CELL_MIN, int(min(w,h)/12))
    cols = max(1, math.ceil(w / cell)); rows = max(1, math.ceil(h / cell))
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, max(0, x//cell))
        cy = min(rows-1, max(0, y//cell))
        grid[cy][cx] += 1
    thr = max(2, int(len(points) / (6*max(1,min(cols,rows)))))
    hits=[(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
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
    regs=[]
    for x0,y0,w0,h0 in rects:
        nx=max(0,x0-10); ny=max(0,y0-10); nw=min(w-nx, w0+20); nh=min(h-ny, h0+20)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

# ---------- analyze region (safe) ----------
def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"row_runs":[],"runs":[]}
    coords = [(p[0], p[1]) for p in pts]; labels=[p[2] for p in pts]
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    bins = max(1, min(12, int(max(1,w/60))))
    col_idx=[]; edges=[i*(w/bins) for i in range(bins+1)]
    for xv in xs:
        ci = 0
        for i in range(bins):
            if edges[i] <= xv <= edges[i+1]:
                ci = i; break
        col_idx.append(ci)
    col_count = max(1, max(col_idx)+1)
    rbins = max(3, min(14, int(max(1,h/ROW_BIN_H))))
    redges=[i*(h/rbins) for i in range(rbins+1)]
    row_idx=[]
    for yv in ys:
        ri=0
        for i in range(rbins):
            if redges[i] <= yv <= redges[i+1]:
                ri=i; break
        row_idx.append(ri)
    row_count = max(1, max(row_idx)+1)
    grid=[['' for _ in range(col_count)] for __ in range(row_count)]
    for i,lbl in enumerate(labels):
        try:
            rix=int(row_idx[i]); cix=int(col_idx[i])
            if 0<=rix<row_count and 0<=cix<col_count:
                grid[rix][cix] = lbl
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
                curc=v; curlen = 1 if v else 0
            if curlen > maxh: maxh = curlen
        row_runs.append(maxh)
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >=4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
            has_multirow=True; break
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"row_runs":row_runs,"runs":runs}

# ---------- overall classification ----------
def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow = sum(1 for b in board_stats if b.get('has_multirow',False))
    if super_count >=1 and long_count >=2 and (super_count + long_count) >=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if multirow >=3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if board_stats and sparse >= len(board_stats)*0.6:
        return "收割时段（胜率调低）", long_count, super_count, multirow
    return "胜率中等", long_count, super_count, multirow

# ---------- annotate ----------
def annotate_pil(pil, regions, stats):
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.load_default()
    except:
        f = None
    for i,r in enumerate(regions):
        x,y,w,h = r
        d.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        s = stats[i]
        txt = f"#{i+1} {s['category']} run={s['maxRun']} multi={s['has_multirow']}"
        d.text((x+4,y+4), txt, fill=(255,255,0), font=f)
    return pil

# ---------- slider interaction helpers ----------
def try_drag_handle(page, handle_locator, dx=260, attempts=1):
    """给定 locator，尝试用鼠标抓住中心并水平拖动 dx 像素"""
    try:
        box = handle_locator.bounding_box()
        if not box:
            return False
        cx = box["x"] + box["width"]/2
        cy = box["y"] + box["height"]/2
        page.mouse.move(cx, cy)
        page.mouse.down()
        step = int(abs(dx)/20) if abs(dx)>0 else 5
        for s in range(1, 21):
            nx = cx + dx * (s/20)
            page.mouse.move(nx, cy, steps=step)
            time.sleep(0.02)
        page.mouse.up()
        return True
    except Exception as e:
        log(f"try_drag_handle fail: {e}")
        return False

def attempt_solve_slider(page, timeout=12):
    """尝试在 page + frames 中寻找常见滑块并拖动，多策略尝试。
       返回 True 如果看起来页面通过验证（由外部逻辑判断实际是否进入桌面）"""
    start = time.time()
    tried = []
    # repeated attempts within timeout
    while time.time() - start < timeout:
        # scan current page + frames for slider-like elements
        candidates = []
        try:
            # search in main frame (page) and all frames
            frames = [page] + page.frames
            for f in frames:
                # selectors to try
                sels = [
                    "[role=slider]",
                    "input[type=range]",
                    ".slider-handle",
                    ".ant-slider-handle",
                    ".drag-handle",
                    ".sliderBtn",
                    ".dragger",
                    ".slider",
                    ".vaptcha-slide-btn",
                    ".geetest_slider_button",
                    ".nc_iconfont.btn_slide"
                ]
                for s in sels:
                    try:
                        loc = f.locator(s)
                        if loc.count() > 0:
                            candidates.append((f, s, loc))
                    except Exception:
                        pass
                # also find any element with draggable attribute
                try:
                    loc2 = f.locator("[draggable='true']")
                    if loc2.count() > 0:
                        candidates.append((f, "[draggable='true']", loc2))
                except Exception:
                    pass
        except Exception as e:
            log(f"frame scan failed: {e}")
        # try candidates in order
        for (frame_ref, selector, loc) in candidates:
            try:
                # pick first visible handle
                for idx in range(min(3, loc.count())):
                    try:
                        handle = loc.nth(idx)
                        if not handle.is_visible():
                            continue
                        key = (selector, idx)
                        if key in tried:
                            continue
                        tried.append(key)
                        log(f"尝试滑块: {selector} (idx {idx})")
                        # strategy 1: drag handle horizontally
                        ok = try_drag_handle(frame_ref, handle, dx=300)
                        time.sleep(0.8)
                        if ok:
                            log("滑动操作已尝试")
                            return True
                    except Exception as e:
                        log(f"candidate try fail: {e}")
                        continue
            except Exception as e:
                log(f"candidate outer fail: {e}")
                continue
        # strategy 2: try JS to set range inputs
        try:
            setrange = page.eval_on_selector_all("input[type=range]", "els => { for (let e of els) e.value = e.max || 100; return els.length }")
            if setrange and int(setrange) > 0:
                log("通过 JS 设置 range inputs")
                return True
        except Exception:
            pass
        # wait a bit and retry
        time.sleep(1.2)
    return False

# ---------- capture screenshot with slider solving ----------
def capture_with_slider():
    """使用 Playwright：访问 DG，点击 Free，尝试解决滑块，截图并返回 (pil, img_bgr)"""
    if not HAVE_PLAY:
        log("Playwright 未装载，无法抓取")
        return None, None, "no_play"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
            ctx = browser.new_context(viewport={"width":1280,"height":900})
            page = ctx.new_page()
            last_error = None
            for url in DG_LINKS:
                try:
                    log(f"打开 {url}")
                    page.goto(url, timeout=30000)
                    time.sleep(1.0)
                    # try clicking many variants of "Free"
                    free_texts = ["Free","免费试玩","免费","Play Free","试玩","进入"]
                    for t in free_texts:
                        try:
                            loc = page.locator(f"text={t}")
                            if loc.count() > 0:
                                log(f"点击按钮: {t}")
                                loc.first.click(timeout=3000)
                                time.sleep(0.8)
                                break
                        except Exception:
                            pass
                    # wait a little for popups
                    time.sleep(1.2)
                    # attempt to detect if we're blocked by slider: screenshot & detect beads quickly
                    shot = page.screenshot(full_page=True)
                    pil = pil_from_bytes(shot)
                    img_bgr = cv_from_pil(pil)
                    pts = detect_beads(img_bgr)
                    # if points small (<= 6) assume didn't reach table; try slider solving
                    if len(pts) < 12:
                        log(f"初次截图点数={len(pts)}，尝试解决滑块/安全条")
                        solved = attempt_solve_slider(page, timeout=12)
                        if solved:
                            log("滑块尝试已执行，等待页面变化")
                            time.sleep(2.2)
                            # take another shot
                            shot2 = page.screenshot(full_page=True)
                            pil2 = pil_from_bytes(shot2)
                            img_bgr2 = cv_from_pil(pil2)
                            pts2 = detect_beads(img_bgr2)
                            log(f"滑块后点数={len(pts2)}")
                            # return the more recent screenshot
                            try:
                                ctx.close()
                            except:
                                pass
                            try:
                                browser.close()
                            except:
                                pass
                            return pil2, img_bgr2, "ok"
                        else:
                            log("未检测到滑块或滑块尝试失败，返回当前截图")
                            try:
                                ctx.close()
                            except: pass
                            try:
                                browser.close()
                            except: pass
                            return pil, img_bgr, "noslider"
                    else:
                        log(f"初次截图点数足够 ({len(pts)})，认为已进入实盘桌面")
                        try:
                            ctx.close()
                        except: pass
                        try:
                            browser.close()
                        except: pass
                        return pil, img_bgr, "ok"
                except Exception as e:
                    last_error = str(e); log(f"访问 {url} 过程异常: {e}")
                    continue
            # all urls failed
            try:
                ctx.close()
            except: pass
            try:
                browser.close()
            except: pass
            return None, None, f"all_url_fail: {last_error}"
    except Exception as e:
        log(f"Playwright outer error: {e}")
        return None, None, f"play_err: {e}"

# ---------- main run ----------
def main():
    log("FINAL_SLIDER_V1 RUN")
    pil, img_bgr, status = capture_with_slider()
    if pil is None:
        log(f"未抓取到页面（status={status}）")
        send_tg_msg(f"⚠️ DG 抓图失败: {status}")
        return
    h,w = img_bgr.shape[:2]
    pts = detect_beads(img_bgr)
    log(f"最终截图点数: {len(pts)}")
    regions = cluster_boards(pts, w, h)
    log(f"聚类桌区: {len(regions)}")
    board_stats = []
    for r in regions:
        try:
            st = analyze_region(img_bgr, r)
        except Exception as e:
            st = {"total":0,"maxRun":0,"category":"error","has_multirow":False,"row_runs":[],"runs":[]}
        board_stats.append(st)
    overall, lcount, scount, mcount = classify_overall(board_stats)
    now = datetime.now(TZ).isoformat()
    summary = {"ts": now, "status": status, "overall": overall, "long_count": lcount, "super_count": scount, "multirow_count": mcount, "boards": board_stats[:40]}
    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # annotate & send screenshot
    pil_ann = annotate_pil(pil, regions, board_stats)
    bio = BytesIO(); pil_ann.save(bio, format="JPEG", quality=85); bio.seek(0)
    caption = f"DG 检测: {overall} (status={status})\n长龙:{lcount} 超龙:{scount} 连珠:{mcount}\n时间:{now}"
    ok = send_tg_photo(bio.read(), caption=caption)
    if ok:
        log("已发送带注释截图到 Telegram")
    else:
        log("发送截图失败")
    # if matches remindable states then also send highlight message
    if overall in ("放水时段（提高胜率）", "中等胜率（中上）"):
        emoji = "🟢" if overall.startswith("放水") else "🔵"
        send_tg_msg(f"{emoji} <b>{overall}</b>\n开始: {now}\n长龙:{lcount} 超龙:{scount} 连珠桌:{mcount}")
    log("运行结束")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("主流程异常: " + str(e))
        log(traceback.format_exc())
        try:
            send_tg_msg(f"⚠️ DG 监测脚本异常: {e}")
        except:
            pass
        sys.exit(0)
