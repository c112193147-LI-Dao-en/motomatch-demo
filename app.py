import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re

# --- 1. 網頁設定 ---
st.set_page_config(
    page_title="MotoMatch AI - 智慧購車顧問", 
    page_icon="🛵", 
    layout="wide"
)

# --- 2. 縣市白名單 (防止亂輸入 ssss) ---
taiwan_cities = [
    "台北市", "新北市", "桃園市", "台中市", "台南市", "高雄市", 
    "基隆市", "新竹市", "嘉義市", "新竹縣", "苗栗縣", "彰化縣", 
    "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", 
    "台東縣", "澎湖縣", "金門縣", "連江縣",
    "台北", "新北", "桃園", "台中", "台南", "高雄", "基隆", "新竹", "嘉義"
]

# --- 3. 數據紀錄函數 ---
def log_action(action_type, details):
    if st.session_state.get('cookie_consent', False):
        log_file = "user_behavior_logs.csv"
        log_data = {
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "location": [st.session_state.get('chat_data', {}).get('location', 'Unknown')],
            "action": [action_type],
            "details": [details],
            "budget": [st.session_state.get('chat_data', {}).get('budget', 0)]
        }
        log_df = pd.DataFrame(log_data)
        if not os.path.isfile(log_file):
            log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
        else:
            log_df.to_csv(log_file, mode='a', index=False, header=False, encoding='utf-8-sig')

# --- 4. 讀取資料 ---
@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
    except:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Style', 'Model', 'Price', 'Image_URL', 'Shop_Link'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['id'] = df.index
    return df

df = load_data()

# --- 5. 核心演算法 (餘弦相似度) ---
@st.cache_resource
def build_similarity_model(data):
    if len(data) < 2: return np.zeros((len(data), len(data)))
    max_price = data['Price'].max() if data['Price'].max() > 0 else 1
    price_norm = data[['Price']] / max_price
    brands_ohe = pd.get_dummies(data['Brand']) * 0.5
    styles_ohe = pd.get_dummies(data['Style']) * 1.0
    features = np.hstack([price_norm.values, brands_ohe.values, styles_ohe.values])
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    features_normalized = features / norm
    return np.dot(features_normalized, features_normalized.T)

# --- 6. 初始化 Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？"}]
if 'chat_stage' not in st.session_state: st.session_state.chat_stage = 0
if 'chat_data' not in st.session_state: st.session_state.chat_data = {}
if 'liked_cars' not in st.session_state: st.session_state.liked_cars = []
if 'last_clicked_car' not in st.session_state: st.session_state.last_clicked_car = None
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = False

# --- 7. Cookie 同意機制 ---
if not st.session_state.cookie_consent:
    with st.container():
        st.warning("🍪 **數據分析授權聲明**")
        st.caption("本系統會記錄匿名行為以優化推薦。點擊按鈕代表同意。")
        if st.button("我同意並開啟購車顧問"):
            st.session_state.cookie_consent = True
            st.rerun()

# --- 8. 左側系統設定去而復返 ---
with st.sidebar:
    st.title("📍 系統設定")
    selected_region = st.selectbox("所在分店", ["全台分店"] + sorted(list(df['Store'].unique() if not df.empty else [])))
    st.divider()
    liked_count = len(st.session_state.liked_cars)
    with st.expander(f"❤️ 我的關注清單 ({liked_count})", expanded=True):
        if liked_count == 0:
            st.caption("尚未收藏車輛")
        else:
            for i, car in enumerate(st.session_state.liked_cars):
                st.markdown(f"**{car['Model']}**")
                if st.button("❌ 移除", key=f"del_{car['id']}"):
                    st.session_state.liked_cars.pop(i)
                    st.rerun()

# --- 9. 主介面 ---
st.title("🛵 MotoMatch AI 智慧導購")
tab1, tab2, tab3 = st.tabs(["💬 AI 購車顧問", "🏠 現場庫存", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: AI 購車顧問 (完整 5 步驟對話)
# ==========================================
with tab1:
    if st.button("🔄 重製對話"):
        st.session_state.chat_stage = 0
        st.session_state.chat_data = {}
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您居住在哪個縣市？"}]
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    stage = st.session_state.chat_stage
    if prompt := st.chat_input("請輸入..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Step 1: 縣市
        if stage == 0:
            if any(city in prompt for city in taiwan_cities):
                st.session_state.chat_data['location'] = prompt
                response = f"收到，您在 **{prompt}**。(2/5) 預算上限是多少？(2萬~12萬)"
                st.session_state.chat_stage = 1
            else:
                response = "📍 抱歉，我不認識這個縣市。請重新輸入（例如：高雄）。"
        # Step 2: 預算
        elif stage == 1:
            try:
                # 先清理字串
                clean = prompt.replace('萬', '0000').replace(',', '').replace(' ', '')
                nums = re.findall(r'\d+', clean)
                if not nums:
                    raise ValueError
                temp_budget = int(nums[0])
                if temp_budget <= 150:  # 使用者輸入 2~150 (代表 2萬~15萬)
                    final_budget = temp_budget * 10000
                else:
                    final_budget = temp_budget # 使用者輸入 20000 以上
                if 20000 <= final_budget <= 150000:
                    st.session_state.chat_data['budget'] = final_budget
                    response = f"預算設定為 **${final_budget:,.0f}**。接下來 (3/5) 主要用途是？(例如：通勤、外送)"
                    st.session_state.chat_stage = 2
                else:
                # 💡 針對「輸入 1」或「輸入太小/太大」的具體引導
                    response = "💰 預算範圍不正確。請輸入 **2萬至12萬** 之間的金額（例如：6萬 或 80000）。"
            except:
                response = "🔢 請輸入有效的數字金額（例如：7萬）。"
        # Step 3: 用途
        elif stage == 2:
            if any(k in prompt for k in ["不知道", "隨便", "沒想法", "不確定"]):
                response = "沒關係！一般二手機車最常用於 **通勤**、**外送** 或 **學生代步**。您覺得哪一個比較貼近您的需求？"
            else:
                st.session_state.chat_data['usage'] = prompt
                response = "(4/5) 好的。接下來，您需要 **ABS 防鎖死煞車系統** 嗎？(提升雨天安全性)"
                st.session_state.chat_stage = 3
        # Step 4: ABS
        elif stage == 3:
            # 偵測使用者是否在詢問知識
            if any(k in prompt for k in ["什麼是", "不懂", "不知", "科普", "差別","甚麼是","?"]):
                response = """🛡️ **MotoBot 小百科：為什麼要選 ABS？**"
                    1. **防打滑**：下雨天急煞時，ABS 能防止輪胎鎖死，避免「撇輪」摔車。 
                    2. **保命符**：在緊急狀況下，它能讓你邊煞車邊轉向閃避障礙物。
                    3. **更安心**：對於新手或通勤族，這是一項能大幅提升安全性的關鍵配備。

                    (4/5) 聽完介紹後，您覺得您的愛車**需要配備 ABS** 嗎？"""
            else:
                # 判斷使用者最終意圖
                st.session_state.chat_data['abs'] = any(k in prompt for k in ["要", "需", "有", "是", "配備"])
                response = "(5/5) 收到。最後，您願意支付 $1500 的跨店調車運費嗎？"
                st.session_state.chat_stage = 4
            
        # Step 5: 運費與結案
        elif stage == 4:
            st.session_state.chat_stage = 5
            response = "🎉 分析完成！根據您的預算、用途與對 ABS 的需求，推薦如下："
            log_action("AI_SEARCH", f"Budget:{st.session_state.chat_data.get('budget')}")

        if stage != 5:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    if st.session_state.chat_stage == 5:
        st.divider()
        budget_limit = st.session_state.chat_data.get('budget', 120000)
        res_df = df[df['Price'] <= budget_limit]
        # 如果使用者要 ABS，過濾掉名稱沒寫 ABS 的車
        if st.session_state.chat_data.get('abs'):
            res_df = res_df[res_df['Model'].str.contains("ABS", case=False, na=False)]
        
        res_df = res_df.head(6)
        if not res_df.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(res_df.iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.write(f"**{row['Model']}**")
                        if st.link_button("👉 查看詳情", row['Shop_Link']):
                            log_action("VIEW", row['Model'])
        else:
            st.warning("😢 找不到完全吻合的車款。")

# ==========================================
# Tab 2 & 3 與 Footer (保持先前所有功能)
# ==========================================
with tab2:
    current_df = df[df['Store'] == selected_region] if selected_region != "全台分店" else df
    cols = st.columns(3)
    for i, (_, row) in enumerate(current_df.head(12).iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.image(row['Image_URL'], use_container_width=True)
                st.write(f"**{row['Model']}**")
                if st.button("❤️ 關注", key=f"lk_{row['id']}"):
                    if row['id'] not in [c['id'] for c in st.session_state.liked_cars]:
                        st.session_state.liked_cars.append(row.to_dict())
                        st.session_state.last_clicked_car = row.to_dict()
                        log_action("LIKE", row['Model'])
                        st.success(f"已關注 {row['Model']}")

with tab3:
    if not st.session_state.liked_cars: st.info("💡 請先關注感興趣的車輛。")
    else:
        target = st.session_state.last_clicked_car or st.session_state.liked_cars[-1]
        sim_model = build_similarity_model(df)
        idx = df[df['id'] == target['id']].index[0]
        scores = sorted(list(enumerate(sim_model[idx])), key=lambda x: x[1], reverse=True)[1:7]
        st.write(f"根據您關注的 **{target['Model']}**，推薦：")
        cols = st.columns(3)
        for i, (s_idx, score) in enumerate(scores):
            r = df.iloc[s_idx]
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(r['Image_URL'], use_container_width=True)
                    st.caption(f"🧬 相似度 {int(score*100)}%")
                    st.markdown(f"**{r['Model']}**")
                    if st.link_button("查看車輛", r['Shop_Link'], use_container_width=True):
                        log_action("VIEW", r['Model'])

st.divider()
with st.expander("⚖️ 責任歸屬界定與免責聲明 [產學合作技術展示]"):
    st.markdown("""
    <div style="font-size: 0.85rem; color: #64748b; line-height: 1.8;">
    1. <b>數據合規：</b> 本系統在獲得授權後匿名記錄行為數據。<br>
    2. <b>資訊準確：</b> 庫存資料以 <b>貳輪嶼門市現場</b> 為準。<br>
    3. <b>責任界定：</b> 本平台為媒合工具，不參與交易，亦不負擔任何交易糾紛責任。
    </div>
    """, unsafe_allow_html=True)    