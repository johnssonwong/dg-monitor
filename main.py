import requests
from PIL import Image
import pytesseract
import os

# --- 配置（可通过环境变量覆盖） ---
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()

DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"

# ✅ 检测截图中的“好路”
def analyze_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='eng')

    # ✅ 检测关键字 “5 row”, “6 row”, “7 B row”, “RowB”, “RowP”, “VRow”
    keywords = ["5 row", "6 row", "7", "8", "9", "RowB", "RowP", "VRow"]
    found = [kw for kw in keywords if kw in text]

    if len(found) >= 3:
        return True, found
    return False, found

# ✅ Telegram推送
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.json()

# ✅ 主程式
def main(image_path):
    matched, matches = analyze_image(image_path)
    if matched:
        message = f"✅ 放水时段出现！好路 ≥3 桌\n\n触发关键词：{', '.join(matches)}\n\n👉 立即留意追龙机会（根据你策略）"
        send_telegram_alert(message)
        print("✅ 通知已发送至 Telegram")
    else:
        print("⛔️ 暂未满足入场条件")

# ✅ 测试方式：上传截图后执行（手动输入路径）
if __name__ == "__main__":
    image_path = "dg_table.png"  # 改成你的图片路径或上传截图名
    main(image_path)
