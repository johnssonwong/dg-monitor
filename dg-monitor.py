import time
import requests
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

# ==== 配置区（已帮你填好） ====
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TIMEZONE_OFFSET = 8  # 马来西亚 UTC+8

# ==== 发送 Telegram 消息 ====
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

# ==== 分析 DG 走势逻辑 ====
def analyze_tables(page):
    # TODO: 这里实现你的 状况A/状况B + 放水/中等胜率 判定
    # 目前假设直接检测到放水
    return {
        "status": "放水时段（提高胜率）",
        "expected_end": datetime.utcnow() + timedelta(minutes=10)
    }

# ==== 主检测逻辑 ====
def run_detection():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 访问 DG
        page.goto(DG_LINKS[0], timeout=60000)
        time.sleep(3)

        # 点击 “免费试玩”
        try:
            page.click("text=免费试玩")
        except:
            try:
                page.click("text=Free")
            except:
                pass
        time.sleep(3)

        # 滚动页面
        page.mouse.wheel(0, 300)
        time.sleep(2)

        # 进入大厅检测
        result = analyze_tables(page)
        browser.close()

        # 发提醒
        if result:
            now = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)
            end_time = result["expected_end"] + timedelta(hours=TIMEZONE_OFFSET)
            remaining = int((end_time - now).total_seconds() / 60)
            send_telegram(
                f"【DG检测提醒】\n状态：{result['status']}\n预计结束时间：{end_time.strftime('%H:%M')}\n剩余：{remaining} 分钟"
            )

if __name__ == "__main__":
    run_detection()
