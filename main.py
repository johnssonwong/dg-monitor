# main.py
# DG 监测脚本（方案 A）
# - 使用 Playwright 注入你已验证的 cookie（DG_COOKIES_JSON）
# - 进入页面后检测是否已进入实盘（通过页面截图中“珠点”数量判断）
# - 如果进入实盘，进行简化版桌区扫描并按你规则判定“放水 / 中等中上 / 不提醒”
# - 发送 Telegram 消息与注释截图
# - 不执行也不教任何绕过滑块的操作；若仍在登录/验证页会把截图发到 Telegram 并提示你更新 cookie

import os, sys, time, json, traceback, math
from datetime import datetime, timezone, timedelta
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except Exception as e:
    print("Playwright 未安装或导入失败：", e)
    raise

# ---------- 配置 ----------
TZ = timezone(timedelta(hours=8))           # 馬來西亞時區 UTC+8
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "").strip()
DG_COOKIES_JSON = os.environ.get("DG_COOKIES_JSON", "").strip()  # 期望为 JSON 数组字符串
# 判定阈值（可按需调整）
POINTS_THRESH_FOR_REAL_TABLE = 12   # 认为进入实盘：截图中检测到 >= 此数量的“珠点”
# 你之前的规则阈值
MIN_BOARDS_FOR_PAW = 3   # 放水至少 3 张桌子满足长龙/超长龙/连珠 等
MID_LONG_REQ = 2         # 中等胜率需要（2 张长龙 + 连珠等）

# ---------- 工具函数 ----------
def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

def send_tg_text(msg):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("TG 未配置，跳过 send msg")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode":"HTML"}, timeout=15)
        return r.ok
    except Exception as e:
        log("send_tg_text error: "+str(e))
        return False

def send_tg_photo_bytes(bytes_img, caption=""):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("TG 未配置，跳过 send photo")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    try:
        files = {"photo": ("shot.jpg", bytes_img)}
        data = {"chat_id": TG_CHAT_ID, "caption": caption, "parse_mode":"HTML"}
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok
    except Exception as e:
        log("send_tg_photo_bytes error: "+str(e)); return False

def pil_to_bytes(img_pil):
    bio = BytesIO(); img_pil.save(bio, format="JPEG", quality=85); bio.seek(0); return bio.read()

# ---------- 图像与珠点检测（基于颜色，鲁棒） ----------
def detect_beads_bgr(img_bgr):
    """
    返回 list of (x,y,label) ， label: 'B' for banker(red), 'P' for player(blue)
    注：颜色阈值可根据你页面微调
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 红色范围（两段）与蓝色范围（简化）
    red_ranges = [((0,100,60),(8,255,255)), ((160,80,60),(179,255,255))]
    blue_range = ((90,60,40),(140,255,255))
    mask_r = None
    for lo,hi in red_ranges:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask_r = part if mask_r is None else cv2.bitwise_or(mask_r, part)
    mask_b = cv2.inRange(hsv, np.array(blue_range[0]), np.array(blue_range[1]))
    # 去噪
    kernel = np.ones((3,3), np.uint8)
    if mask_r is not None:
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)
    points=[]
    for mask, label in ((mask_r,'B'), (mask_b,'P')):
        if mask is None: continue
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 6: continue
            M = cv2.moments(c)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points

# ---------- 简单板块切分 & 区域分析（稳健实现，避免 index error） ----------
def split_grid_regions(w,h,cell=160):
    cols = max(1, w // cell)
    rows = max(1, h // (cell//3))  # 使行数多一点
    regs=[]
    cw = w / cols; ch = h / rows
    for r in range(rows):
        for c in range(cols):
            x = int(c*cw); y = int(r*ch)
            ww = int(cw); hh = int(ch)
            # 保证在图内
            if x+ww > w: ww = w-x
            if y+hh > h: hh = h-y
            regs.append((x,y,ww,hh))
    return regs

def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts = detect_beads_bgr(crop)
    # flatten simple run detection: sort by x then y to simulate列优先
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False}
    pts_sorted = sorted(pts, key=lambda p:(p[0], p[1]))
    flat = [p[2] for p in pts_sorted]
    # compute max consecutive same
    maxRun=1; cur=flat[0]; ln=1
    for v in flat[1:]:
        if v==cur: ln+=1
        else:
            maxRun = max(maxRun, ln)
            cur=v; ln=1
    maxRun = max(maxRun, ln)
    if maxRun >= 10: cat = "超长龙"
    elif maxRun >= 8: cat = "长龙"
    elif maxRun >= 4: cat = "长连"
    elif maxRun == 1: cat = "单跳"
    else: cat = "双跳/短连"
    # detect multirow (粗略)：按 y 分三段，如果三段都各自出现 >=4 连则认为有多连/连珠
    h_third = max(1, h//3)
    rows_ok = 0
    for i in range(3):
        yy = i*h_third; hh = h_third if i<2 else (h - 2*h_third)
        subcrop = crop[yy:yy+hh,:,:]
        pts_sub = detect_beads_bgr(subcrop)
        if pts_sub:
            # quick check max run inside this sub
            xs = sorted(pts_sub, key=lambda p:p[0]); f = [p[2] for p in xs]
            m=1;cur=f[0];ln=1
            for v in f[1:]:
                if v==cur: ln+=1
                else:
                    m=max(m,ln); cur=v; ln=1
            m=max(m,ln)
            if m>=4: rows_ok+=1
    has_multi = rows_ok >= 3
    return {"total": len(pts), "maxRun": maxRun, "category": cat, "has_multirow": has_multi}

# ---------- 整体判定规则（实现你指定的逻辑） ----------
def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('长龙','超长龙'))
    super_count = sum(1 for b in board_stats if b['category']=='超长龙')
    multirow_count = sum(1 for b in board_stats if b.get('has_multirow',False))
    # 放水判定：符合至少 3 张桌子的长龙/超长龙/多连条件
    if (super_count + long_count) >= 3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    # 中等中上：至少有 3 张桌子出现 多连/连珠（三行连续多连）且至少有 2 张长龙/超长龙
    if multirow_count >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    # 否则
    return "胜率中等", long_count, super_count, multirow_count

# ---------- 注释图片 ----------
def annotate_image(pil_img, regions, board_stats):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except:
        font=None
    for r,st in zip(regions, board_stats):
        x,y,w,h = r
        draw.rectangle([x,y,x+w,y+h], outline=(255,0,0), width=2)
        label = f"{st['category']} run={st['maxRun']} pts={st['total']}"
        draw.text((x+4, y+4), label, fill=(255,255,0), font=font)
    return pil_img

# ---------- 主要流程 ----------
def main():
    log("脚本开始")
    # 解析 cookie
    cookies = []
    if DG_COOKIES_JSON:
        try:
            cookies = json.loads(DG_COOKIES_JSON)
            log(f"读取到 {len(cookies)} 个 cookie")
        except Exception as e:
            send_tg_text("⚠️ DG_COOKIES_JSON 解析失败，请检查 Secrets 格式（必须为 JSON 数组字符串）。")
            log("cookie 解析失败: "+str(e))
            return
    else:
        send_tg_text("⚠️ 未找到 DG_COOKIES_JSON（请把你已验证的 cookie JSON 放到仓库 Secrets）。")
        return

    # 启动 Playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
            context = browser.new_context(viewport={"width":1366,"height":900})
            # 注入 cookie（注意 Playwright 的 cookie 需要 domain、name、value）
            try:
                context.add_cookies(cookies)
                log("已注入 cookie 到浏览器 context")
            except Exception as e:
                log("注入 cookie 失败: "+str(e))
            page = context.new_page()
            entered=False
            for url in DG_LINKS:
                try:
                    log(f"打开：{url}")
                    page.goto(url, timeout=30000)
                    time.sleep(0.8)
                    # 尝试点击 Free
                    for label in ["Free","免费试玩","免费","试玩","进入"]:
                        try:
                            loc = page.locator(f"text={label}")
                            if loc.count() > 0:
                                log(f"点击按钮: {label}")
                                loc.first.click(timeout=4000)
                                time.sleep(0.8)
                                break
                        except Exception:
                            pass
                    # 截图并检测珠点数
                    shot = page.screenshot(full_page=True)
                    pil = Image.open(BytesIO(shot)).convert("RGB")
                    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    pts = detect_beads_bgr(img)
                    log(f"截图检测到珠点：{len(pts)}")
                    send_tg_photo_bytes(pil_to_bytes(pil), caption=f"初始截图 points={len(pts)} 时间:{nowstr()}")
                    if len(pts) >= POINTS_THRESH_FOR_REAL_TABLE:
                        entered=True
                        # 进入实盘，进行区域检测
                        h,w = img.shape[:2]
                        regions = split_grid_regions(w,h,cell=180)
                        board_stats=[]
                        for r in regions:
                            st = analyze_region(img, r)
                            board_stats.append(st)
                        overall, lc, sc, mc = classify_overall(board_stats)
                        # 估算“放水/结束时间” —— 仅为启发式估算（基于 maxRun）
                        max_runs = [b['maxRun'] for b in board_stats]
                        max_run_overall = max(max_runs) if max_runs else 0
                        # 启发式：每个连续粒估计 0.8~2 分钟，取 1.2 分钟为基线
                        est_total_minutes = int(max(1, min(180, math.ceil(max_run_overall * 1.2))))
                        est_remain_minutes = est_total_minutes  # 简单把当前剩余近似设为估算时长（不可保证，提示透明）
                        caption = f"判定: {overall}  (长龙:{lc} 超龙:{sc} 连珠桌:{mc})\n估算持续: {est_total_minutes} 分钟（启发式）\n时间:{nowstr()}"
                        ann = annotate_image(pil.copy(), regions, board_stats)
                        send_tg_photo_bytes(pil_to_bytes(ann), caption=caption)
                        # 只在两种需要提醒的时段发送提醒文本
                        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
                            emoji = "🟢" if overall.startswith("放水") else "🔵"
                            send_tg_text(f"{emoji} <b>{overall}</b>\n时间: {nowstr()}\n长龙:{lc} 超龙:{sc} 连珠桌:{mc}\n估算剩余: {est_remain_minutes} 分钟（启发式）")
                        else:
                            log("未达到提醒时段（属于胜率中等或收割时段），不发送入场提醒。")
                        break
                    else:
                        # 未探测到实盘珠点 -> 可能仍在登录/验证页
                        send_tg_text("⚠️ 似乎未进入 DG 实盘（points 未达阈值）。请手动在浏览器完成 Free->滚动安全条一次，并把会话 cookie 更新到 Secrets（DG_COOKIES_JSON）。")
                        # 另外把当前页面截图发上来，便于你检查
                        send_tg_photo_bytes(pil_to_bytes(pil), caption="当前页面截图（可能是登录/验证页）")
                        break
                except PlaywrightTimeout:
                    log("页面访问超时，尝试下一个链接或重试")
                    continue
                except Exception as e:
                    log("页面流程异常: "+str(e))
                    send_tg_text("⚠️ 脚本访问页面异常: "+str(e))
                    send_tg_photo_bytes(pil_to_bytes(Image.open(BytesIO(page.screenshot(full_page=True))).convert("RGB")), caption="异常时截图")
                    break
            # 结束清理
            try:
                page.close()
                context.close()
                browser.close()
            except:
                pass
            if not entered:
                log("本次未进入实盘")
    except Exception as e:
        log("主流程异常: "+str(e))
        send_tg_text("⚠️ 脚本主流程异常: " + str(e) + "\n请检查 Actions 日志。")
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
    sys.exit(0)
