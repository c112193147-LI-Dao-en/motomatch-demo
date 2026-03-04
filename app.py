import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import time
from streamlit_javascript import st_javascript
# 機器學習核心套件 (請確保 requirements.txt 有 scikit-learn)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- 1. 網頁設定與數據載入 ---
st.set_page_config(page_title="MotoMatch AI - 智慧導購系統", page_icon="🛵", layout="wide")

# 台灣縣市白名單 (AI 顧問地理攔截用)
taiwan_cities = ["台北", "新北", "基隆", "桃園", "新竹", "苗栗", "台中", "彰化", "南投", "雲林", "嘉義", "台南", "高雄", "屏東", "宜蘭", "花蓮", "台東", "澎湖", "金門", "連江"]

@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
        def augment_data(row):
            m = str(row['Model']).upper()
            # 💡 精準規格判定邏輯 (修正 MT 與 R15) [cite: 2026-01-01, 2026-01-02]
            abs_keys = ["ABS", "GR", "MMBCU", "DRG", "AUGUR", "MT-", "R15", "V4", "V3", "XMAX"]
            tcs_keys = ["TCS", "MMBCU", "DRG", "R15M", "V4", "AUGUR", "MT-09", "XMAX"]
            row['ABS_VAL'] = 1 if any(x in m for x in abs_keys) else 0
            row['TCS_VAL'] = 1 if any(x in m for x in tcs_keys) else 0
            
            brand_map = {"山葉": ["YAMAHA", "R15", "MT"], "三陽": ["SYM", "DRG", "JET"], "光陽": ["KYMCO"]}
            row['Brand'] = "其他"
            for b, k in brand_map.items():
                if any(x.upper() in m for x in k): row['Brand'] = b; break
            return row
            
        df = df.apply(augment_data, axis=1)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['id'] = df.index
        return df
    except:
        return pd.DataFrame(columns=['id', 'Store', 'Brand', 'Model', 'Price', 'Image_URL', 'Shop_Link', 'ABS_VAL', 'TCS_VAL'])

df = load_data()

# --- 2. 核心推薦演算法：餘弦相似度 (基於瀏覽行為) ---
def get_cosine_recs(history, full_df, top_n=6):
    if not history or full_df.empty: return pd.DataFrame()
    features = ['Price', 'ABS_VAL', 'TCS_VAL']
    data_matrix = full_df[features].copy()
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(data_matrix)
    hist_ids = [c['id'] for c in history]
    hist_indices = full_df[full_df['id'].isin(hist_ids)].index
    if len(hist_indices) == 0: return pd.DataFrame()
    # 建立興趣向量：取平均值
    user_profile = np.mean(scaled_matrix[hist_indices], axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_profile, scaled_matrix).flatten()
    full_df = full_df.copy()
    full_df['score'] = similarities
    return full_df[~full_df['id'].isin(hist_ids)].sort_values(by='score', ascending=False).head(top_n)

# --- 3. 初始化 Session State ---
if 'view_history' not in st.session_state: st.session_state.view_history = []
if 'liked_cars' not in st.session_state: st.session_state.liked_cars = []
if 'cookie_synced' not in st.session_state: st.session_state.cookie_synced = False
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'chat_stage' not in st.session_state: st.session_state.chat_stage = 0
if 'chat_data' not in st.session_state: st.session_state.chat_data = {}
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。請問您居住在哪個縣市？"}]

# --- 4. 持久化 Cookie 讀取 ---
js_read = "parent.localStorage.getItem('moto_history');"
stored_data = st_javascript(js_read)
if stored_data and stored_data != "null" and not st.session_state.cookie_synced:
    try:
        current_history = json.loads(stored_data)
        if isinstance(current_history, list) and len(current_history) > 0:
            st.session_state.view_history = current_history
            st.session_state.cookie_synced = True 
            st.rerun()
    except: pass

def save_to_cookie(history_list):
    json_str = json.dumps(history_list).replace("'", "\\'")
    js_write = f"parent.localStorage.setItem('moto_history', '{json_str}');"
    st_javascript(js_write)

# --- 5. 主介面標籤頁 ---
st.title("🛵 MotoMatch AI 智慧導購")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI 顧問", "🏠 現場庫存", "🔮 猜你喜歡", "⚖️ 規格對照", "🕒 最近瀏覽"])

# --- Tab 1: AI 顧問 (完整對話邏輯補回) [cite: 2025-12-28, 2026-01-08] ---
with tab1:
    st.info("🤖 **操作提示**：請回答 MotoBot 的問題，我將為您過濾合適的機車庫存。")
    if st.button("🔄 重製對話"):
        st.session_state.chat_stage = 0; st.session_state.chat_data = {}
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。請問您居住在哪個縣市？"}]
        st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    stage = st.session_state.chat_stage
    if stage < 5 and (prompt := st.chat_input("請輸入回答...", key="chat_input_field")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if stage == 0:
            user_loc = next((city for city in taiwan_cities if city in prompt), None)
            if user_loc:
                st.session_state.chat_data['location'] = user_loc; st.session_state.chat_stage = 1
                response = f"收到，您在 **{user_loc}**。(2/5) 預算上限是多少？"
            else: response = "📍 抱歉，目前的服務僅限台灣地區。請輸入正確的台灣縣市名稱。"
        
        elif stage == 1:
            try:
                nums = re.findall(r'\d+', prompt.replace('萬', '0000'))
                val = int(nums[0]); final_budget = val * 10000 if val < 200 else val
                st.session_state.chat_data['budget'] = final_budget; st.session_state.chat_stage = 2
                response = f"預算已設定為 **${final_budget:,.0f}**。(3/5) 主要用途是？"
            except: response = "🔢 請輸入有效的數字金額。"
            
        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt; st.session_state.chat_stage = 3
            response = "(4/5) 需要 **ABS 安全系統** 嗎？(不懂可以問：什麼是 ABS？)"
            
        elif stage == 3:
            if any(k in prompt for k in ["什麼", "不懂", "是啥"]):
                response = "🛡️ **MotoBot 小百科**：ABS 能防止急煞時輪胎鎖死。您覺得需要嗎？"
            else:
                st.session_state.chat_data['abs'] = any(k in prompt for k in ["要", "是", "需"])
                st.session_state.chat_stage = 4
                response = "(5/5) 若心儀車款在其他縣市，願意付 $1500 跨店調車費嗎？"
        
        elif stage == 4:
            st.session_state.chat_data['shipping'] = not any(n in prompt for n in ["不", "否", "沒"])
            st.session_state.chat_stage = 5; response = "🎉 分析完成！推薦結果如下："
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    if stage == 5:
        u_budget = st.session_state.chat_data.get('budget', 150000)
        rec_df = df[df['Price'] <= u_budget]
        if not st.session_state.chat_data.get('shipping', True):
            rec_df = rec_df[rec_df['Store'].str.contains(st.session_state.chat_data.get('location', ''), na=False)]
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(rec_df.head(6).iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(row['Image_URL'], use_container_width=True)
                    st.write(f"**{row['Model']}**")
                    st.subheader(f"💰 ${int(row['Price']):,}")

# --- Tab 2: 🏠 現場庫存 (分頁加固) ---
with tab2:
    st.success("🎯 **快速上手**：點擊車款下方的 「❤️ 關注」，系統將會記憶您的喜好。")
    s1, s2 = st.columns([2, 1])
    with s1: search_query = st.text_input("🔍 快速找車", placeholder="輸入型號關鍵字...")
    with s2: selected_region = st.sidebar.selectbox("所在分店", ["全台分店"] + sorted(list(df['Store'].unique())))
    
    f_df = df.copy()
    if selected_region != "全台分店": f_df = f_df[f_df['Store'] == selected_region]
    if search_query: f_df = f_df[f_df['Model'].str.contains(search_query, case=False, na=False)]
    
    if f_df.empty: st.warning("找不到符合車款。")
    else:
        items_per_page = 12
        total_pages = max(1, (len(f_df) - 1) // items_per_page + 1)
        if st.session_state.current_page > total_pages: st.session_state.current_page = 1
        page_df = f_df.iloc[(st.session_state.current_page-1)*12 : st.session_state.current_page*12]
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(page_df.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(row['Image_URL'], use_container_width=True)
                    st.subheader(f"💰 ${int(row['Price']):,}")
                    st.write(f"**{row['Model']}**")
                    c1, c2 = st.columns(2)
                    if c1.button("❤️ 關注", key=f"fav_btn_{row['id']}"):
                        if row.to_dict() not in st.session_state.liked_cars: st.session_state.liked_cars.append(row.to_dict())
                        if row.to_dict() not in st.session_state.view_history:
                            st.session_state.view_history.append(row.to_dict())
                            save_to_cookie(st.session_state.view_history); time.sleep(0.3)
                        st.rerun()
                    c2.link_button("🌐 網站", row.get('Shop_Link', '#'), use_container_width=True)
        
        # 分頁按鈕優化
        st.divider()
        pages = range(1, total_pages + 1)
        p_btns = st.columns(min(len(pages), 12))
        for idx, p in enumerate(list(pages)[:12]):
            if p_btns[idx].button(f"{'★' if p == st.session_state.current_page else ''}{p}", key=f"page_btn_{p}"):
                st.session_state.current_page = p; st.rerun()

# --- Tab 3: 🔮 猜你喜歡 (餘弦相似度) ---
with tab3:
    st.header("🔮 AI 偏好推薦")
    if not st.session_state.view_history: st.info("請先去庫存區點擊 ❤️。")
    else:
        recs = get_cosine_recs(st.session_state.view_history, df)
        r_cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with r_cols[i % 3]:
                with st.container(border=True):
                    st.image(row['Image_URL'], use_container_width=True)
                    st.write(f"**{row['Model']}**")
                    st.subheader(f" ${int(row['Price']):,}")
                    st.caption(f"相似度: {row['score']:.2%}")

# --- Tab 4: ⚖️ 規格對照 ---
with tab4:
    st.header("車款規格比對")
    if len(st.session_state.liked_cars) < 2: st.info("請關注至少 2 台。")
    else:
        display_cars = st.session_state.liked_cars[:4]
        comp_cols = st.columns(len(display_cars))
        for i, car in enumerate(display_cars):
            with comp_cols[i]:
                with st.container(border=True):
                    st.image(car['Image_URL'], use_container_width=True)
                    st.caption(f"**{car['Model']}**")
                    st.markdown(f"{'✅' if car['ABS_VAL'] else '❌'} ABS 安全")
                    st.markdown(f"{'✅' if car['TCS_VAL'] else '❌'} TCS 循跡")
                    if st.button("移除", key=f"rem_btn_{car['id']}"):
                        st.session_state.liked_cars.pop(i); st.rerun()

# --- Tab 5: 🕒 最近瀏覽 ---
with tab5:
    st.header(" 最近瀏覽紀錄")
    if st.button(" 清空紀錄", type="primary"):
        st.session_state.view_history = []; save_to_cookie([]); time.sleep(0.5); st.rerun()
    v_cols = st.columns(5)
    for i, car in enumerate(reversed(st.session_state.view_history)):
        with v_cols[i % 5]:
            with st.container(border=True):
                st.image(car['Image_URL'], use_container_width=True)
                st.caption(car['Model'])