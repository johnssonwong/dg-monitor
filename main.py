# main.py
# DG 专用监测（强化版）— 带 geetest canvas 模板匹配滑块求位移与真人式拖动
# 使用说明：
# - 复制此文件覆盖仓库 main.py
# - 在 workflow 中保留 Playwright 安装与 chromium 安装步骤
# - Secrets: TG_BOT_TOKEN, TG_CHAT_ID
# - 运行后会把 debug 图片发到 Telegram（初始、滑块处理所用图片/差异/最终注释图）

import os, sys, time, json, traceback, base64, math, random
from datetime import datetime, timezone, timedelta
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PLAY = True
except Exception:
    HAVE_PLAY = False

# ---------- config ----------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))
POINTS_THRESH_FOR_REAL_TABLE = 12

# image color thresholds (大体)
RED_RANGES = [((0,100,70),(8,255,255)), ((160,80,70),(179,255,255))]
BLUE_RANGE = ((90,60,50),(140,255,255))
MIN_CONTOUR_AREA = 8

def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

def send_tg_msg(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置：跳过 send msg")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text, "parse_mode":"HTML"}, timeout=20)
        return r.ok
    except Exception as e:
        log("send_tg_msg fail: "+str(e)); return False

def send_tg_photo(bytes_img, caption=""):
    if not TG_TOKEN or not TG_CHAT:
        log("TG 未配置：跳过 send photo"); return False
    try:
        files = {"photo": ("shot.jpg", bytes_img)}
        data = {"chat_id": TG_CHAT, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto", files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log("send_tg_photo fail: "+str(e)); return False

def pil_to_bytes(pil):
    bio=BytesIO(); pil.save(bio, format="JPEG", quality=85); bio.seek(0); return bio.read()

def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---------- bead detection ----------
def detect_beads(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r = None
    for lo,hi in RED_RANGES:
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
            a = cv2.contourArea(c)
            if a < MIN_CONTOUR_AREA: continue
            M = cv2.moments(c)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx,cy,lbl))
    return pts

# ---------- region cluster & analyze (简洁版，保证稳定) ----------
def cluster_boards(points, w, h):
    if not points: return []
    cell = max(60, int(min(w,h)/12))
    cols = max(1, (w+cell-1)//cell); rows = max(1, (h+cell-1)//cell)
    grid = [[0]*cols for _ in range(rows)]
    for x,y,_ in points:
        gx = min(cols-1, max(0, x//cell)); gy = min(rows-1, max(0, y//cell))
        grid[gy][gx] += 1
    thr = max(2, int(len(points) / (6*max(1, min(cols,rows)))))
    hits=[(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
        regs=[]
        for ry in range(rows):
            for rx in range(cols):
                regs.append((int(rx*cell), int(ry*cell), int(cell), int(cell)))
        return regs
    rects=[]
    for r,c in hits:
        x0=r*cell; y0=c*cell
        pass
    # fallback: return coarse grid rectangles
    regs=[]
    for ry in range(rows):
        for rx in range(cols):
            regs.append((int(rx*cell), int(ry*cell), int(cell), int(cell)))
    return regs

def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False}
    xs = [p[0] for p in pts]; ys=[p[1] for p in pts]; labels=[p[2] for p in pts]
    # collapse into flattened by scanning columns left->right then rows top->bottom (粗略)
    cols = max(1, int(w/60))
    rows = max(1, int(h/28))
    grid = [['' for _ in range(cols)] for __ in range(rows)]
    for i,lbl in enumerate(labels):
        cx,cy = xs[i], ys[i]
        ci = min(cols-1, int(cx/(w/cols))) if w>0 else 0
        ri = min(rows-1, int(cy/(h/rows))) if h>0 else 0
        grid[ri][ci] = lbl
    flattened=[]
    for c in range(cols):
        for r in range(rows):
            v = grid[r][c]
            if v: flattened.append(v)
    runs=[]; maxRun=0
    if flattened:
        cur=flattened[0]; ln=1
        for v in flattened[1:]:
            if v==cur: ln+=1
            else:
                runs.append((cur,ln)); maxRun=max(maxRun,ln); cur=v; ln=1
        runs.append((cur,ln)); maxRun=max(maxRun,ln)
    cat="other"
    if maxRun>=10: cat="super_long"
    elif maxRun>=8: cat="long"
    elif maxRun>=4: cat="longish"
    elif maxRun==1: cat="single"
    # detect multirow (3行连开)
    has_multirow=False
    # 如果横向分区里某 3 个相邻行都 >=4
    # 这里简单用 rows 上的 run info
    row_runs=[]
    for r in range(rows):
        maxr=0; cur=None; ln=0
        for c in range(cols):
            v = grid[r][c]
            if v and v==cur: ln+=1
            elif v:
                cur=v; ln=1
            else:
                cur=None; ln=0
            if ln>maxr: maxr=ln
        row_runs.append(maxr)
    for i in range(0,len(row_runs)-2):
        if row_runs[i]>=4 and row_runs[i+1]>=4 and row_runs[i+2]>=4:
            has_multirow=True; break
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow}

def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow = sum(1 for b in board_stats if b.get('has_multirow',False))
    if super_count>=1 and long_count>=2 and (super_count+long_count)>=3:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if (long_count + super_count) >= 3:
        return "放水时段（提高胜率）", long_count, super_count, multirow
    if multirow >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow
    return "胜率中等", long_count, super_count, multirow

# ---------- geetest helpers ----------
def b64_to_cv2(b64data):
    header, data = b64data.split(",",1) if "," in b64data else ("",b64data)
    imgdata = base64.b64decode(data)
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def find_gap_by_template(full_img, bg_img):
    # convert to gray, compute diff or template match
    try:
        fgray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        bgray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(fgray, bgray)
        # blur reduce noise
        diff = cv2.GaussianBlur(diff, (5,5), 0)
        # threshold
        _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # sum columns
        col_sum = np.sum(th, axis=0)
        # find column with max sum -> gap center
        x = int(np.argmax(col_sum))
        return x, diff
    except Exception as e:
        log("find_gap error: "+str(e))
        return None, None

def perform_slider_drag(frame, slider_handle, distance_px):
    # segmented human-like drag
    try:
        box = slider_handle.bounding_box()
        if not box:
            return False
        start_x = box['x'] + box['width']/2
        start_y = box['y'] + box['height']/2
        frame.mouse.move(start_x, start_y)
        frame.mouse.down()
        total = distance_px
        moved = 0
        steps = max(6, int(abs(total)/10))
        for i in range(steps):
            # proportion with overshoot/compensation
            frac = (i+1)/steps
            # ease out movement
            target = start_x + total * (1 - (1-frac)**2)
            jitter = random.uniform(-2,2)
            frame.mouse.move(target, start_y + random.uniform(-1,1) + jitter, steps=1)
            time.sleep(0.02 + random.uniform(0,0.03))
        # small backward correction
        frame.mouse.move(start_x + total - random.uniform(4,8), start_y, steps=2)
        time.sleep(0.08)
        frame.mouse.up()
        return True
    except Exception as e:
        log("perform_slider_drag fail: "+str(e)); return False

# ---------- capture & solver ----------
def capture_and_solve():
    if not HAVE_PLAY:
        log("Playwright 未安装"); send_tg_msg("⚠️ Playwright 未安装，无法运行"); return None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width":1366,"height":900},
                                      user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = context.new_page()
        status_note = "unknown"
        try:
            for url in DG_LINKS:
                try:
                    log("open "+url)
                    page.goto(url, timeout=30000)
                    time.sleep(0.8)
                    # attempt click Free / 免费试玩
                    for t in ["Free","免费试玩","免费","Play Free","试玩","进入"]:
                        try:
                            loc = page.locator(f"text={t}")
                            if loc.count() > 0:
                                log("click "+t)
                                loc.first.click(timeout=4000)
                                time.sleep(0.8)
                                break
                        except Exception:
                            pass
                    # initial screenshot
                    shot1 = page.screenshot(full_page=True)
                    pil1 = pil_from_bytes(shot1); img1 = cv_from_pil(pil1)
                    pts1 = detect_beads(img1)
                    log(f"初始点数 {len(pts1)}")
                    send_tg_photo(pil_to_bytes(pil1), caption=f"初始截图 points={len(pts1)}")
                    if len(pts1) >= POINTS_THRESH_FOR_REAL_TABLE:
                        status_note = "ok-entered-initial"
                        # close resources
                        context.close(); browser.close()
                        return pil1, img1, status_note
                    # try find geetest canvases in page + frames
                    solved_any = False
                    frames = [page] + list(page.frames)
                    for f in frames:
                        try:
                            # try canvas selectors
                            selectors = [".geetest_canvas_fullbg", ".geetest_canvas_bg", ".geetest_fullbg", ".geetest_bg"]
                            canvases = {}
                            for sel in selectors:
                                try:
                                    if f.locator(sel).count() > 0:
                                        # try to get toDataURL from first matched element
                                        try:
                                            data = f.eval_on_selector(sel, "el => el.toDataURL()")
                                            canvases[sel] = data
                                        except Exception:
                                            # maybe the image is background-image (css)
                                            try:
                                                css = f.eval_on_selector(sel, "el => window.getComputedStyle(el).backgroundImage")
                                                if css and "url(" in css:
                                                    imgurl = css.split("url(")[1].split(")")[0].strip('\"\'')
                                                    # fetch remote/base64
                                                    r = requests.get(imgurl, timeout=15)
                                                    canvases[sel] = "data:image/jpeg;base64," + base64.b64encode(r.content).decode()
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                            # also try to find slider handle
                            slider_loc = None
                            possible_handles = [".geetest_slider_button", ".geetest_slider_btn", ".geetest_widget_button"]
                            for sh in possible_handles:
                                try:
                                    if f.locator(sh).count() > 0:
                                        slider_loc = f.locator(sh).first
                                        break
                                except Exception:
                                    pass
                            if canvases and slider_loc:
                                # try to get fullbg & bg images
                                full_key = None; bg_key=None
                                for k in canvases.keys():
                                    if "full" in k: full_key=k
                                    elif "bg" in k: bg_key=k
                                if not full_key:
                                    # sometimes alt names
                                    keys = list(canvases.keys())
                                    if len(keys)>=1: full_key = keys[0]
                                if not bg_key and len(canvases)>=2:
                                    keys = list(canvases.keys())
                                    bg_key = keys[1] if keys[0]==full_key and len(keys)>1 else None
                                if not full_key:
                                    continue
                                full_b64 = canvases.get(full_key)
                                bg_b64 = canvases.get(bg_key) if bg_key else None
                                if not bg_b64:
                                    # some geetest provide sliced bg only in DOM as img; try to read .geetest_slice or .geetest_canvas_slice
                                    try:
                                        if f.locator(".geetest_canvas_slice").count() > 0:
                                            bg_b64 = f.eval_on_selector(".geetest_canvas_slice", "el => el.toDataURL()")
                                    except Exception:
                                        pass
                                if not bg_b64:
                                    # fallback: we will try to use fullbg vs page screenshot (less accurate)
                                    bg_img = None
                                else:
                                    full_img = b64_to_cv2(full_b64)
                                    bg_img = b64_to_cv2(bg_b64)
                                    xgap, diff = find_gap_by_template(full_img, bg_img)
                                    if xgap is None:
                                        log("找不到 gap")
                                        continue
                                    # compute scale: canvas width vs slider track width
                                    # find slider bounding box
                                    sb = slider_loc.bounding_box()
                                    if not sb:
                                        continue
                                    canvas_w = full_img.shape[1]
                                    # estimate distance in px on screen ~ (xgap / canvas_w) * track_pixel_width
                                    # find track element width: try ".geetest_slider_track" or use slider handle parent width
                                    track_w = None
                                    try:
                                        if f.locator(".geetest_slider_track").count() > 0:
                                            tb = f.locator(".geetest_slider_track").first.bounding_box()
                                            track_w = tb['width']
                                        else:
                                            parent = f.eval_on_selector("body", "el => 0")  # dummy
                                    except Exception:
                                        pass
                                    if not track_w:
                                        # approximate track width as page viewport width * 0.6
                                        track_w = page.viewport_size['width'] * 0.6 if page.viewport_size else sb['width']*8
                                    # compute target move
                                    move_px = (xgap / float(canvas_w)) * track_w
                                    # adjust calibration (empirical)
                                    move_px = max(30, move_px - 6)
                                    log(f"Found geetest gap x={xgap}, canvas_w={canvas_w}, track_w={track_w}, move_px={move_px}")
                                    # debug: send full/bg/diff images
                                    try:
                                        send_debug_images(full_img, bg_img, diff, caption="geetest: full/bg/diff")
                                    except Exception:
                                        pass
                                    # perform segmented drag on slider_loc
                                    ok = perform_slider_drag(f, slider_loc, move_px)
                                    time.sleep(1.2)
                                    # after drag check if points increased
                                    shot_after = page.screenshot(full_page=True)
                                    pil_after = pil_from_bytes(shot_after); img_after = cv_from_pil(pil_after)
                                    pts_after = detect_beads(img_after)
                                    log(f"drag后 points={len(pts_after)}")
                                    send_tg_photo(pil_to_bytes(pil_after), caption=f"滑块尝试后 points={len(pts_after)}")
                                    if len(pts_after) >= POINTS_THRESH_FOR_REAL_TABLE:
                                        log("成功进入实盘（点数阈值）")
                                        context.close(); browser.close()
                                        return pil_after, img_after, "ok-after-drag"
                                    # else maybe need retry: small back/forth
                                    # do small retries
                                    for attempt in range(2):
                                        perform_slider_drag(f, slider_loc, move_px * (0.92 + random.uniform(-0.05,0.05)))
                                        time.sleep(1.0 + random.uniform(0,0.6))
                                        shot_retry = page.screenshot(full_page=True)
                                        pil_r = pil_from_bytes(shot_retry); img_r = cv_from_pil(pil_r)
                                        pts_r = detect_beads(img_r)
                                        log(f"retry points={len(pts_r)}")
                                        if len(pts_r) >= POINTS_THRESH_FOR_REAL_TABLE:
                                            context.close(); browser.close()
                                            return pil_r, img_r, "ok-after-retry"
                                    # otherwise continue scanning frames
                        except Exception as e:
                            log("frame solve error: "+str(e))
                            continue
                    # after scanning frames and attempts, if still not in table, return current screenshot
                    context.close(); browser.close()
                    return pil1, img1, "not-entered"
                except Exception as e:
                    log("page try error: "+str(e)); continue
        except Exception as e:
            log("capture error: "+str(e))
            try:
                context.close()
            except: pass
            try:
                browser.close()
            except: pass
            return None
    return None

def send_debug_images(full_img, bg_img, diff_img, caption=""):
    # convert cv images to PIL and send three images concatenated for convenience
    try:
        fpil = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
        bpil = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
        dpil = Image.fromarray(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
        # create combined
        w = fpil.width + bpil.width + dpil.width
        h = max(fpil.height, bpil.height, dpil.height)
        comb = Image.new("RGB", (w,h), (20,20,20))
        comb.paste(fpil, (0,0)); comb.paste(bpil, (fpil.width,0)); comb.paste(dpil, (fpil.width+bpil.width,0))
        send_tg_photo(pil_to_bytes(comb), caption=caption)
    except Exception as e:
        log("send_debug_images fail: "+str(e))

# ---------- main ----------
def main():
    try:
        if not HAVE_PLAY:
            log("Playwright 未就绪"); send_tg_msg("⚠️ Playwright 未就绪"); return
        res = capture_and_solve()
        if not res:
            send_tg_msg("⚠️ 抓取/滑块处理未能完成，请查看 Actions 日志。")
            return
        pil, img, status = res
        # detect boards and classify (简化流程)
        pts = detect_beads(img)
        h,w = img.shape[:2]
        regions = cluster_boards(pts, w, h)
        board_stats=[]
        for r in regions:
            st = analyze_region(img, r)
            board_stats.append(st)
        overall, lc, sc, mc = classify_overall(board_stats)
        summary = {"ts": datetime.now(TZ).isoformat(), "status": status, "overall": overall,
                   "long_count": lc, "super_count": sc, "multirow_count": mc, "boards": board_stats[:40]}
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
        # annotate final screenshot
        ann = pil.copy()
        draw = ImageDraw.Draw(ann)
        try:
            font = ImageFont.load_default()
        except:
            font=None
        for i,r in enumerate(regions):
            x,y,w0,h0=r
            draw.rectangle([x,y,x+w0,y+h0], outline=(255,0,0), width=2)
            st = board_stats[i] if i<len(board_stats) else {}
            draw.text((x+4,y+4), f"#{i+1} {st.get('category','?')} run={st.get('maxRun',0)}", fill=(255,255,0), font=font)
        send_tg_photo(pil_to_bytes(ann), caption=f"最终判定: {overall} status={status}")
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            emoji = "🟢" if overall.startswith("放水") else "🔵"
            send_tg_msg(f"{emoji} <b>{overall}</b>\n时间:{nowstr()}\n长龙:{lc} 超龙:{sc} 连珠:{mc}")
        log("完成")
    except Exception as e:
        log("主流程异常: "+str(e)); log(traceback.format_exc())
        try:
            send_tg_msg("⚠️ 脚本异常: "+str(e))
        except:
            pass

if __name__ == "__main__":
    main()
    # ensure success exit
    sys.exit(0)
