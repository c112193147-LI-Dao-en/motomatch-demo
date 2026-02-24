import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re

# --- 1. 網頁設定 ---
st.set_page_config(page_title="MotoMatch AI - 智慧購車顧問", page_icon="🛵", layout="wide")

# --- 2. 縣市與分店白名單 (產學邏輯核心) ---
store_cities = ["新北", "桃園", "新竹", "台中", "台南", "高雄", "花蓮"]
taiwan_cities = store_cities + ["台北", "基隆", "嘉義", "苗栗", "彰化", "南投", "雲林", "屏東", "宜蘭", "台東", "澎湖", "金門", "連江"]

# --- 3. 數據紀錄函數 (產學數據收集) ---
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
        try:
            log_df = pd.DataFrame(log_data)
            log_df.to_csv(log_file, mode='a', index=False, header=not os.path.isfile(log_file), encoding='utf-8-sig')
        except:
            pass

# --- 4. 讀取與處理資料 ---
@st.cache_data 
def load_data():
    try:
        # 讀取你提供的標記後資料
        df = pd.read_csv("labeled_data.csv")
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['id'] = df.index
        return df
    except:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Style', 'Model', 'Price', 'Image_URL', 'Shop_Link'])

df = load_data()

# --- 5. 相似度演算法 (推薦系統核心) ---
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
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = False

# --- 7. Cookie 與側邊欄 ---
if not st.session_state.cookie_consent:
    with st.container():
        st.warning("🍪 **數據分析授權聲明**")
        st.caption("本系統會記錄匿名行為以優化推薦並供專案分析。點擊按鈕代表同意授權。")
        if st.button("我同意並開啟購車顧問"):
            st.session_state.cookie_consent = True
            st.rerun()

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
                if st.button("❌ 移除", key=f"sidebar_del_{car['id']}"):
                    st.session_state.liked_cars.pop(i)
                    st.rerun()

# --- 8. 主介面 ---
st.title("🛵 MotoMatch AI 智慧導購")
tab1, tab2, tab3 = st.tabs(["💬 AI 購車顧問", "🏠 現場庫存", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: AI 購車顧問 (含防呆與科普)
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
    if stage < 5 and (prompt := st.chat_input("請輸入...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Step 1: 縣市判定
        if stage == 0:
            if any(city in prompt for city in taiwan_cities):
                st.session_state.chat_data['location'] = prompt[:2] 
                st.session_state.chat_stage = 1
                response = f"收到，您在 **{prompt}**。(2/5) 預算上限是多少？(2萬~15萬)"
            else:
                response = "📍 抱歉，我不認識這個縣市。請重新輸入。"

        # Step 2: 經費換算
        elif stage == 1:
            try:
                clean = prompt.replace('萬', '0000').replace(',', '').replace(' ', '')
                nums = re.findall(r'\d+', clean)
                temp_budget = int(nums[0])
                final_budget = temp_budget * 10000 if temp_budget <= 150 else temp_budget
                if 20000 <= final_budget <= 150000:
                    st.session_state.chat_data['budget'] = final_budget
                    st.session_state.chat_stage = 2
                    response = f"預算設定為 **${final_budget:,.0f}**。接下來 (3/5) 主要用途是？"
                else:
                    response = "💰 預算範圍不正確。請輸入 2萬至15萬 之間的金額。"
            except:
                response = "🔢 請輸入有效的數字金額。"

        # Step 3: 用途
        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt
            st.session_state.chat_stage = 3
            response = "(4/5) 好的。接下來，您需要 **ABS 防鎖死煞車系統** 嗎？"

        # Step 4: ABS 科普
        elif stage == 3:
            if any(k in prompt for k in ["什麼是", "不懂", "不知", "科普", "差別", "?", "甚麼是"]):
                response = """🛡️ **MotoBot 小百科：為什麼要選 ABS？**
1. **防打滑**：雨天急煞時防止輪胎鎖死，避免摔車。
2. **保命符**：緊急狀況下仍能邊煞車邊轉向。
(4/5) 聽完介紹後，您覺得需要配備 ABS 嗎？"""
            else:
                st.session_state.chat_data['abs'] = any(k in prompt for k in ["要", "是", "需", "有", "配備"])
                st.session_state.chat_stage = 4
                response = "(5/5) 最後，您願意支付 $1500 的運費進行跨店調車嗎？"

        # Step 5: 跨店意願與南投攔截
        elif stage == 4:
            negatives = ["不", "沒", "否", "拒絕", "不想"]
            is_negative = any(n in prompt for n in negatives)
            positives = ["願意", "好", "可以", "是", "要"]
            st.session_state.chat_data['shipping_ready'] = any(p in prompt for p in positives) and not is_negative
            st.session_state.chat_stage = 5
            st.session_state.messages = [{"role": "assistant", "content": "🎉 分析完成！推薦結果如下："}]
            log_action("AI_SEARCH", f"Budget:{st.session_state.chat_data.get('budget')}")
            st.rerun()

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # --- 顯示 AI 推薦結果 (地理物理隔離) ---
    if st.session_state.chat_stage == 5:
        st.divider()
        u_loc = st.session_state.chat_data.get('location', '')
        u_ship = st.session_state.chat_data.get('shipping_ready', False)
        u_budget = st.session_state.chat_data.get('budget', 120000)
        u_abs = st.session_state.chat_data.get('abs', False)

        final_df = pd.DataFrame()
        if u_ship:
            final_df = df[df['Price'] <= u_budget]
        else:
            has_local = any(u_loc in city for city in store_cities)
            if has_local:
                final_df = df[(df['Price'] <= u_budget) & (df['Store'].str.contains(u_loc, na=False))]
            else:
                st.error(f"📍 抱歉，目前 **{u_loc}** 地區尚無直營分店。")
                st.info("💡 由於您不願意跨店調車，建議點擊「重製對話」並選擇『願意』以查看全台分店。")

        if not final_df.empty and u_abs:
            final_df = final_df[final_df['Model'].str.contains("ABS", case=False, na=False)]

        if not final_df.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(final_df.head(6).iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.write(f"**{row['Model']}**")
                        st.caption(f"📍 門市：{row['Store']}")
                        if st.link_button("👉 查看詳情", row['Shop_Link']):
                            log_action("VIEW", row['Model'])

# ==========================================
# Tab 2: 現場庫存 (優化關注穩定性)
# ==========================================
with tab2:
    current_df = df[df['Store'] == selected_region] if selected_region != "全台分店" else df
    cols = st.columns(3)
    for i, (_, row) in enumerate(current_df.head(12).iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.image(row['Image_URL'], use_container_width=True)
                st.write(f"**{row['Model']}**")
                if st.button("❤️ 關注", key=f"lk_tab2_{row['id']}"):
                    if row['id'] not in [c['id'] for c in st.session_state.liked_cars]:
                        st.session_state.liked_cars.append(row.to_dict())
                        log_action("LIKE", row['Model'])
                        st.success(f"已加入關注清單")
                        st.rerun()

# ==========================================
# Tab 3: 猜你喜歡 (優化版面與提示)
# ==========================================
with tab3:
    if not st.session_state.liked_cars:
        st.info("💡 **探索靈感**：目前您的關注清單是空的。請先點擊 ❤️ 關注您感興趣的車輛，我將為您精準推薦！")
    else:
        target = st.session_state.liked_cars[-1]
        st.subheader(f"🔮 根據您對「{target['Model']}」的興趣推薦：")
        st.divider()

        sim_model = build_similarity_model(df)
        idx = df[df['id'] == target['id']].index[0]
        scores = sorted(list(enumerate(sim_model[idx])), key=lambda x: x[1], reverse=True)[1:7]

        cols = st.columns(3)
        for i, (s_idx, score) in enumerate(scores):
            r = df.iloc[s_idx]
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**🔥 契合度 {int(score*100)}%**")
                    st.image(r['Image_URL'], use_container_width=True)
                    st.markdown(f"**{r['Model']}**")
                    st.write(f"💰 ${r['Price']:,.0f} | 📍 {r['Store']}")
                    if st.link_button("👉 查看車輛", r['Shop_Link'], use_container_width=True):
                        log_action("REC_VIEW", r['Model'])

st.divider()
with st.expander("⚖️ 免責聲明"):
    st.markdown("1. 數據合規：本系統僅紀錄匿名行為供產學分析。 2. 資訊準確：實體車況以門市現場為準。 3. 責任界定：本平台僅供購車決策參考之媒合工具。")