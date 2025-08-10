# detector.py
# DG 自动检测脚本（用于 GitHub Actions 每次运行一次）
# 复制到仓库根目录，配合上面的 workflow 使用

import os, sys, json, time, datetime, subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import requests
import asyncio
from playwright.async_api import async_playwright

# --------- 参数（如需要可在此微调） ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "485427847")
DG_URLS = [os.environ.get("DG_URL_1","https://dg18.co/wap/"), os.environ.get("DG_URL_2","https://dg18.co/")]
STATE_FILE = Path("state.json")
SCREENSHOT = "screen.png"

# 图像检测阈值（可按你的页面分辨率微调）
MIN_TABLE_AREA = 2500
DRAGON_PIXEL_V = 10    # 竖直像素连续高度近似当作长龙的阈值（启发式）
SUPER_DRAGON_PIXEL_V = 12
MIN_DRAGON_TABLES = 3
PERCENT_FULL_THRESHOLD = 0.5  # ≥50% 桌面为放水规则一

# ---------------- Telegram ----------------
def send_text(msg, image_path=None):
    base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode":"HTML"}
    try:
        r = requests.post(base + "/sendMessage", data=payload, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("sendMessage failed:", e)
    if image_path and Path(image_path).exists():
        try:
            files = {"photo": open(image_path,"rb")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": msg}
            r = requests.post(base + "/sendPhoto", data=data, files=files, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print("sendPhoto failed:", e)

# ---------------- state utils ----------------
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except:
            pass
    return {"in_water": False, "start_ts": None, "last_seen": None, "in_mid_high": False}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    gh = os.environ.get("GITHUB_TOKEN")
    if not gh:
        return
    try:
        subprocess.run(["git","config","user.email","actions@github.com"], check=False)
        subprocess.run(["git","config","user.name","github-actions[bot]"], check=False)
        subprocess.run(["git","add", str(STATE_FILE)], check=False)
        subprocess.run(["git","commit","-m","update state.json from action"], check=False)
        origin = os.environ.get("GITHUB_SERVER_URL","https://github.com") + "/" + os.environ.get("GITHUB_REPOSITORY","")
        if origin:
            push_url = origin.replace("https://", f"https://x-access-token:{gh}@")
            subprocess.run(["git","push", push_url, "HEAD:refs/heads/HEAD"], check=False)
    except Exception as e:
        print("git push failed:", e)

# ---------------- Image analysis helpers ----------------
def find_white_contours(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,200]); upper = np.array([255,40,255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    h,w = img_bgr.shape[:2]
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww*hh < MIN_TABLE_AREA: continue
        rects.append((x,y,ww,hh))
    return rects

def analyze_region(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # red mask (two ranges)
    lower1 = np.array([0,60,40]); upper1 = np.array([12,255,255])
    lower2 = np.array([170,60,40]); upper2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    mask_blue = cv2.inRange(hsv, np.array([90,60,30]), np.array([140,255,255]))
    red = int(np.count_nonzero(mask_red))
    blue = int(np.count_nonzero(mask_blue))
    total = crop.shape[0]*crop.shape[1]
    red_ratio = red/total
    blue_ratio = blue/total
    # compute max vertical run for each mask
    def max_vert_run(mask):
        max_run = 0
        for col in range(mask.shape[1]):
            curr=0; col_max=0
            for v in mask[:,col]:
                if v:
                    curr += 1
                    col_max = max(col_max, curr)
                else:
                    curr = 0
            max_run = max(max_run, col_max)
        return max_run
    red_v = max_vert_run(mask_red)
    blue_v = max_vert_run(mask_blue)
    return {"red_ratio":red_ratio,"blue_ratio":blue_ratio,"red_v":red_v,"blue_v":blue_v}

# ---------------- classify overall using your rules ----------------
def classify_all(analyses):
    total = len(analyses)
    dragon_tables = 0
    super_dragon = 0
    full_score = 0
    for a in analyses:
        if a["red_v"] >= SUPER_DRAGON_PIXEL_V or a["blue_v"] >= SUPER_DRAGON_PIXEL_V:
            super_dragon += 1
            dragon_tables += 1
        elif a["red_v"] >= DRAGON_PIXEL_V or a["blue_v"] >= DRAGON_PIXEL_V:
            dragon_tables += 1
        if (a["red_ratio"] + a["blue_ratio"]) > 0.006:
            full_score += 1
    percent_full = full_score / max(1,total)
    rule1 = percent_full >= PERCENT_FULL_THRESHOLD
    rule2 = (super_dragon >= 1) and ((dragon_tables - super_dragon) >= 2)
    is_water = (rule1 or rule2) and (dragon_tables >= MIN_DRAGON_TABLES)
    is_mid_high = (dragon_tables >= 2) and (not is_water)
    is_low = (dragon_tables < 2) and (percent_full < 0.2)
    is_mid = (not is_water) and (not is_mid_high) and (not is_low)
    return {
        "total": total, "dragon": dragon_tables, "super": super_dragon,
        "full": full_score, "percent_full": percent_full,
        "rule1": rule1, "rule2": rule2,
        "is_water": is_water, "is_mid_high": is_mid_high, "is_mid": is_mid, "is_low": is_low
    }

# ---------------- detect "profitable" tables using状况A启发式 ----------------
def detect_profitable(analyses):
    idxs = []
    for i,a in enumerate(analyses):
        # 超长龙或长龙都纳入考虑；这里用竖直连长度作近似
        if a["red_v"] >= SUPER_DRAGON_PIXEL_V or a["blue_v"] >= SUPER_DRAGON_PIXEL_V:
            idxs.append(i)
        elif a["red_v"] >= DRAGON_PIXEL_V or a["blue_v"] >= DRAGON_PIXEL_V:
            idxs.append(i)
    return idxs

# ---------------- Playwright: open DG and screenshot ----------------
async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"], headless=True)
        context = await browser.new_context(viewport={"width":1280,"height":900})
        page = await context.new_page()
        opened = False
        for url in DG_URLS:
            try:
                await page.goto(url, timeout=45000)
                opened = True
                break
            except Exception as e:
                print("goto failed:", e)
        if not opened:
            raise RuntimeError("cannot open DG URLs")
        await page.wait_for_timeout(2500)
        # click Free / 免费试玩 if present
        selectors = ["text=Free","text=免费试玩","text=免费","button:has-text('Free')","button:has-text('免费')"]
        for s in selectors:
            try:
                el = await page.query_selector(s)
                if el:
                    try:
                        await el.click(timeout=3000)
                        await page.wait_for_timeout(2500)
                        break
                    except:
                        pass
            except:
                pass
        # try some scrolling to trigger content load
        await page.mouse.wheel(0, 800)
        await page.wait_for_timeout(2000)
        await page.mouse.wheel(0, -200)
        await page.wait_for_timeout(4000)
        # final wait for game tiles to render
        await page.wait_for_timeout(4000)
        await page.screenshot(path=SCREENSHOT, full_page=True)
        await context.close()
        await browser.close()

# ---------------- main ----------------
async def main_async():
    now_my = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    try:
        await capture()
    except Exception as e:
        send_text(f"❗ DG monitor: 无法打开 DG 页面或截图失败：{e}")
        return
    # analyze screenshot
    img = cv2.imread(SCREENSHOT)
    if img is None:
        send_text("❗ DG monitor: 截图无法读取")
        return
    rects = find_white_contours(img)
    analyses = []
    for (x,y,w,h) in rects:
        crop = img[y:y+h, x:x+w]
        analyses.append(analyze_region(crop))
    if not analyses:
        # fallback: analyze entire image as one table
        analyses = [ analyze_region(img) ]

    overall = classify_all(analyses)
    profitable_idxs = detect_profitable(analyses)

    # load/save state to mark start/end of 放水
    state = load_state()
    summary = f"检测时间(MYT): {now_my.strftime('%Y-%m-%d %H:%M:%S')}\n总桌: {overall['total']}, 长龙桌: {overall['dragon']}, 超长龙: {overall['super']}, 饱满桌: {overall['full']}\npercent_full:{overall['percent_full']:.2f}\nrule1:{overall['rule1']} rule2:{overall['rule2']}\n"

    # 判定逻辑与提醒
    if overall["is_water"]:
        # 必须提醒：放水开始或持续
        if not state.get("in_water", False):
            state["in_water"] = True
            state["start_ts"] = datetime.datetime.utcnow().isoformat()
            state["last_seen"] = datetime.datetime.utcnow().isoformat()
            save_state(state)
            msg = "🚨 放水时段（提高胜率）检测到！\n" + summary + f"可盈利桌数量: {len(profitable_idxs)}\n请手动入场。"
            send_text(msg, SCREENSHOT)
        else:
            # 已在放水中，仅更新 last_seen（不重复通知）
            state["last_seen"] = datetime.datetime.utcnow().isoformat()
            save_state(state)
            print("放水仍在，更新 last_seen")
    elif overall["is_mid_high"]:
        # 小提醒：中等胜率（中上）
        if not state.get("in_mid_high", False):
            state["in_mid_high"] = True
            state["mid_high_start"] = datetime.datetime.utcnow().isoformat()
            save_state(state)
            msg = "🔔 中等胜率（中上）检测到 — 小提醒。\n" + summary
            send_text(msg, SCREENSHOT)
        else:
            state["mid_high_last"] = datetime.datetime.utcnow().isoformat()
            save_state(state)
            print("中等胜率持续，已更新时间")
    else:
        # 非放水/非中等中上 => 若之前处于放水或中等中上，发送结束通知
        if state.get("in_water", False):
            start = datetime.datetime.fromisoformat(state.get("start_ts"))
            end = datetime.datetime.utcnow()
            dur = end - start
            mins = int(dur.total_seconds() / 60)
            msg = f"✅ 放水已结束。\n开始(MYT): {start.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n结束(MYT): {end.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n持续: {mins} 分钟\n" + summary
            send_text(msg, SCREENSHOT)
            state["in_water"] = False
            state["start_ts"] = None
            state["last_seen"] = None
            save_state(state)
        if state.get("in_mid_high", False):
            state["in_mid_high"] = False
            state["mid_high_start"] = None
            state["mid_high_last"] = None
            save_state(state)

    # 日志到控制台
    print("Overall:", overall)
    print("Profitable indices:", profitable_idxs)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
