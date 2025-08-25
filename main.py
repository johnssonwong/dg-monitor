# main.py
# DG 实盘监测 — 加强版（滑块/iframe 处理 + IndexError 修复 + 不抛出 exit code 1）
# 说明：替换后在 GitHub Actions 里运行（或本地），会把初始截图、滑块尝试截图与最终注释截图发到 Telegram 以便核验。

import os, sys, time, json, traceback, random
from datetime import datetime, timezone, timedelta
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import numpy as np
except Exception:
    np = None

# Playwright
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PLAY = True
except Exception:
    HAVE_PLAY = False

# -------- Config (可按需微调) --------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
SUMMARY_FILE = "last_run_summary.json"
STATE_FILE = "state.json"
TZ = timezone(timedelta(hours=8))

# Image detection params
RED_RANGES = [((0,100,70),(8,255,255)), ((160,80,70),(179,255,255))]
BLUE_RANGE = ((90,60,50),(140,255,255))
MIN_CONTOUR_AREA = 8
CELL_MIN = 60
ROW_BIN_H = 28

# Detection thresholds
POINTS_THRESH_FOR_REAL_TABLE = 12  # 当截图珠点 >= 12 认为是实盘桌面（可微调）
# --------------------------------------

def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

def send_tg_message(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置，跳过 send message")
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
        log("TG 未配置，跳过 send photo")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        files = {"photo": ("shot.jpg", bytes_img)}
        data = {"chat_id": TG_CHAT, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log("send photo fail: " + str(e))
        return False

# PIL <-> OpenCV helpers
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def pil_to_bytes(pil):
    bio = BytesIO(); pil.save(bio, format="JPEG", quality=85); bio.seek(0); return bio.read()

def cv_from_pil(pil):
    arr = np.array(pil) if np else None
    if arr is None:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ---------------- image analysis ----------------
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
            if M.get("m00",0) == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx,cy,lbl))
    return pts

def cluster_boards_safe(points, w, h):
    if not points:
        return []
    cell = max(CELL_MIN, int(min(w,h)/12))
    cols = max(1, (w + cell - 1)//cell)
    rows = max(1, (h + cell - 1)//cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, max(0, x//cell))
        cy = min(rows-1, max(0, y//cell))
        grid[cy][cx]+=1
    thr = max(2, int(len(points) / (6*max(1,min(cols,rows)))))
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
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
        nx=max(0,x0-12); ny=max(0,y0-12); nw=min(w-nx, w0+24); nh=min(h-ny, h0+24)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_region_safe(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads_opencv(crop)
    # normalize to list of tuples (cx,cy,label)
    pts_list = list(pts) if pts else []
    if not pts_list:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"row_runs":[],"runs":[]}
    # get xs ys and labels safely
    xs = [p[0] for p in pts_list]; ys = [p[1] for p in pts_list]; labels = [p[2] for p in pts_list]
    # determine bin counts
    bins = max(1, min(12, int(max(1,w/60))))
    edges = [i*(w/bins) for i in range(bins+1)]
    col_idx = []
    for xv in xs:
        # compute index robustly
        ci = int(min(bins-1, max(0, int((xv / w) * bins)))) if w>0 else 0
        col_idx.append(ci)
    col_count = max(1, max(col_idx)+1)
    rbins = max(3, min(14, int(max(1,h/ROW_BIN_H))))
    row_idx=[]
    for yv in ys:
        ri = int(min(rbins-1, max(0, int((yv / h) * rbins)))) if h>0 else 0
        row_idx.append(ri)
    row_count = max(1, max(row_idx)+1)
    grid = [['' for _ in range(col_count)] for __ in range(row_count)]
    for i,lbl in enumerate(labels):
        try:
            rix = int(row_idx[i]); cix = int(col_idx[i])
            if 0 <= rix < row_count and 0 <= cix < col_count:
                grid[rix][cix] = lbl
        except Exception:
            continue
    flattened=[]
    for c in range(col_count):
        for r in range(row_count):
            v = grid[r][c]
            if v:
                flattened.append(v)
    runs=[]
    if flattened:
        cur = {"color":flattened[0],"len":1}
        for v in flattened[1:]:
            if v == cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur); cur = {"color":v,"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
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
    has_multirow=False
    for i in range(0, max(0, len(row_runs)-2)):
        if row_runs[i] >= 4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
            has_multirow = True; break
    cat="other"
    if maxRun >= 10: cat="super_long"
    elif maxRun >= 8: cat="long"
    elif maxRun >= 4: cat="longish"
    elif maxRun == 1: cat="single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"row_runs":row_runs,"runs":runs}

def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow_count = sum(1 for b in board_stats if b.get('has_multirow',False))
    # rule A: super + 2 long
    if super_count >=1 and long_count >=2 and (super_count + long_count) >=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    if multirow_count >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    if board_stats and sparse >= len(board_stats)*0.6:
        return "收割时段（胜率调低）", long_count, super_count, multirow_count
    return "胜率中等", long_count, super_count, multirow_count

def annotate_and_bytes(pil_img, regions, board_stats):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    for i,r in enumerate(regions):
        x,y,w,h = r
        draw.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        st = board_stats[i] if i < len(board_stats) else {}
        txt = f"#{i+1} {st.get('category','?')} run={st.get('maxRun',0)} multi={st.get('has_multirow',False)}"
        draw.text((x+4, y+4), txt, fill=(255,255,0), font=font)
    return pil_to_bytes(pil_img)

# ------------- Playwright capture + slider attempts -------------
def attempt_drag_on_handle(frame, handle, dx=300, steps=20):
    try:
        box = handle.bounding_box()
        if not box:
            return False
        cx = box["x"] + box["width"]/2
        cy = box["y"] + box["height"]/2
        frame.mouse.move(cx, cy)
        frame.mouse.down()
        for s in range(steps):
            # progressive movement with jitter
            nx = cx + (dx*(s+1)/steps) + random.uniform(-3,3)
            ny = cy + random.uniform(-2,2)
            frame.mouse.move(nx, ny, steps=1)
            time.sleep(0.02 + random.uniform(0,0.03))
        frame.mouse.up()
        return True
    except Exception as e:
        log("attempt_drag_on_handle fail: " + str(e))
        return False

def try_solve_slider_on_page(page, timeout_sec=14):
    """扫描 page + all frames，尝试滑块。返回 True 如果看起来已操作（不保证成功），False 否则"""
    start = time.time()
    tried = set()
    selectors = [
        ".geetest_slider_button",".geetest_canvas_slice",".geetest_canvas_fullbg",".nc_iconfont.btn_slide",
        ".vaptcha-slide-btn",".drag-handle",".slider-handle","[role=slider]","input[type=range]"
    ]
    while time.time() - start < timeout_sec:
        # gather candidates from page and frames
        frames = [page] + list(page.frames)
        for f in frames:
            for sel in selectors:
                try:
                    loc = f.locator(sel)
                    cnt = 0
                    try:
                        cnt = loc.count()
                    except Exception:
                        cnt = 0
                    if cnt > 0:
                        for idx in range(min(3,cnt)):
                            key = (f, sel, idx)
                            if key in tried: continue
                            tried.add(key)
                            handle = loc.nth(idx)
                            if not handle.is_visible():
                                continue
                            log(f"尝试滑块 selector={sel} idx={idx}")
                            # scroll into view
                            try:
                                handle.scroll_into_view_if_needed(timeout=1500)
                            except:
                                pass
                            ok = attempt_drag_on_handle(f, handle, dx=300, steps=18)
                            time.sleep(0.9 + random.uniform(0,0.6))
                            if ok:
                                return True
                except Exception:
                    continue
        # JS approach: set input[type=range]
        try:
            res = page.eval_on_selector_all("input[type=range]", "els => { for (let e of els) { try { e.value = e.max || 100; e.dispatchEvent(new Event('change')); } catch(e){} } return els.length }")
            if res and int(res) > 0:
                log("通过 JS 设置 range 输入")
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False

def capture_page_with_slider():
    if not HAVE_PLAY:
        log("Playwright 未安装或不可用")
        return None, None, "no_playwright"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True,
                                       args=["--no-sandbox","--disable-blink-features=AutomationControlled","--disable-dev-shm-usage"])
            context = browser.new_context(viewport={"width":1366,"height":900},
                                          user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)")
            page = context.new_page()
            last_err = None
            for url in DG_LINKS:
                try:
                    log(f"打开: {url}")
                    page.goto(url, timeout=35000)
                    time.sleep(0.8)
                    # try clicking Free buttons
                    for t in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
                        try:
                            el = page.locator(f"text={t}")
                            if el.count() > 0:
                                log(f"点击: {t}")
                                el.first.click(timeout=4000)
                                time.sleep(0.7)
                                break
                        except Exception:
                            continue
                    # initial screenshot
                    shot1 = page.screenshot(full_page=True)
                    pil1 = pil_from_bytes(shot1); img1 = cv_from_pil(pil1)
                    pts1 = detect_beads_opencv(img1)
                    log(f"初次截图珠点数量: {len(pts1)}")
                    send_tg_photo(pil_to_bytes(pil1), caption=f"初始截图 (points={len(pts1)})")
                    # If few points, attempt slider
                    if len(pts1) < POINTS_THRESH_FOR_REAL_TABLE:
                        log("检测到疑似验证页，尝试滑块/iframe 处理")
                        solved = try_solve_slider_on_page(page, timeout_sec=16)
                        # after attempt, take another screenshot
                        shot2 = page.screenshot(full_page=True)
                        pil2 = pil_from_bytes(shot2); img2 = cv_from_pil(pil2)
                        pts2 = detect_beads_opencv(img2)
                        log(f"滑块尝试后珠点数量: {len(pts2)} (solved_attempt={solved})")
                        send_tg_photo(pil_to_bytes(pil2), caption=f"滑块尝试后截图 (points={len(pts2)})")
                        # decide which screenshot to use
                        final_pil = pil2 if len(pts2) >= len(pts1) else pil1
                        final_img = img2 if len(pts2) >= len(pts1) else img1
                        status = "ok" if len(detect_beads_opencv(final_img)) >= POINTS_THRESH_FOR_REAL_TABLE else "not_entered"
                        try:
                            context.close()
                        except: pass
                        try:
                            browser.close()
                        except: pass
                        return final_pil, final_img, status
                    else:
                        # points sufficient: use initial
                        try:
                            context.close()
                        except: pass
                        try:
                            browser.close()
                        except: pass
                        return pil1, img1, "ok"
                except Exception as e:
                    last_err = str(e); log("页面访问异常: " + str(e))
                    continue
            try:
                context.close()
            except: pass
            try:
                browser.close()
            except: pass
            return None, None, f"all_urls_fail: {last_err}"
    except Exception as e:
        log("Playwright outer exception: " + str(e))
        return None, None, f"playout: {e}"

# ---------------- main logic ----------------
def main_once():
    try:
        pil, img_bgr, status = capture_page_with_slider()
        if pil is None or img_bgr is None:
            send_tg_message(f"⚠️ DG 抓图失败 (status={status})")
            log("抓图失败，结束本次不报错退出")
            return
        h,w = img_bgr.shape[:2]
        points = detect_beads_opencv(img_bgr)
        log(f"最终截图检测到总珠点数: {len(points)}")
        regions = cluster_boards_safe(points, w, h)
        log(f"聚类出 {len(regions)} 个候选桌区")
        board_stats=[]
        for r in regions:
            try:
                st = analyze_region_safe(img_bgr, r)
            except Exception as e:
                log("分析单桌异常: " + str(e))
                st = {"total":0,"maxRun":0,"category":"error","has_multirow":False,"row_runs":[],"runs":[]}
            board_stats.append(st)
        overall, long_c, super_c, multi_c = classify_overall(board_stats)
        summary = {"ts": datetime.now(TZ).isoformat(), "status": status, "overall": overall,
                   "long_count": long_c, "super_count": super_c, "multirow_count": multi_c,
                   "boards": board_stats[:40]}
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # annotate and send final screenshot
        final_bytes = annotate_and_bytes(pil, regions, board_stats)
        cap = f"DG 判定: {overall} (status={status})\n长龙:{long_c} 超龙:{super_c} 连珠:{multi_c}\n时间:{nowstr()}"
        send_tg_photo(final_bytes, caption=cap)
        # alert only for the two desired states
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            emoji = "🟢" if overall.startswith("放水") else "🔵"
            send_tg_message(f"{emoji} <b>{overall}</b>\n开始: {nowstr()}\n长龙:{long_c} 超龙:{super_c} 连珠桌:{multi_c}")
        log("本次检测完成 -> " + overall)
    except Exception as e:
        log("主流程捕获异常: " + str(e))
        log(traceback.format_exc())
        # notify but DO NOT return non-zero exit (to avoid Process completed with exit code 1)
        try:
            send_tg_message(f"⚠️ DG 监测脚本异常: {e}")
        except:
            pass

if __name__ == "__main__":
    # single run (use Actions schedule or workflow to loop)
    try:
        main_once()
    except Exception as e:
        log("顶层异常: " + str(e))
        try:
            send_tg_message(f"⚠️ DG 顶层异常: {e}")
        except:
            pass
    # ensure exit 0 so Actions doesn't show exit code 1 for handled errors
    sys.exit(0)
