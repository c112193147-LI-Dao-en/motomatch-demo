import streamlit as st
import pandas as pd
import numpy as np
import math
import re # 引入正規表達式來抓取數字和關鍵字

# --- 1. 網頁設定 ---
st.set_page_config(
    page_title="MotoMatch", 
    page_icon="🛵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 讀取資料 ---
@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
    except FileNotFoundError:
        return pd.DataFrame() 
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Image_URL'] = df['Image_URL'].fillna('https://cdn-icons-png.flaticon.com/512/3097/3097180.png')
    for col in ['Store', 'Brand', 'Style']:
        if col not in df.columns: df[col] = '未知'
    df['id'] = df.index
    return df

df = load_data()

# --- 3. 核心演算法 (相似度) ---
@st.cache_resource
def build_similarity_model(data):
    if len(data) < 2: return np.zeros((len(data), len(data)))
    max_price = data['Price'].max() if data['Price'].max() > 0 else 1
    price_norm = data[['Price']] / max_price
    brands_ohe = pd.get_dummies(data['Brand']) * 1.5 
    styles_ohe = pd.get_dummies(data['Style']) * 1.2
    features = np.hstack([price_norm.values, brands_ohe.values, styles_ohe.values])
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    features_normalized = features / norm
    cosine_sim = np.dot(features_normalized, features_normalized.T)
    return cosine_sim

# --- 4. 關鍵字分析機器人 (Simulated AI Parser) ---
def parse_user_intent(user_input, all_stores):
    """
    這是一個「模擬 AI」的邏輯函數。
    它不聯網，而是分析使用者打的字來猜測意圖。
    """
    filters = {}
    user_input = user_input.lower() # 轉小寫方便比對

    # 1. 抓預算 (尋找數字)
    # 邏輯：抓出字串中的數字，如果有 "萬"，就乘 10000
    try:
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            budget_raw = int(numbers[0])
            if "萬" in user_input or budget_raw < 100: # 使用者可能打 "4萬" 或 "4"
                filters['budget'] = budget_raw * 10000
            else:
                filters['budget'] = budget_raw # 使用者打 "40000"
    except:
        pass # 沒打數字就算了

    # 2. 抓地點 (比對店家名稱)
    # 邏輯：檢查輸入是否有包含 "高雄", "台中", "新北" 等字眼
    for store in all_stores:
        # 取店名的一部分來比對 (例如 "高雄店" -> 抓 "高雄")
        city_keyword = store.replace("店", "").replace("分", "") 
        if city_keyword in user_input:
            filters['store'] = store
            break
    
    # 3. 抓用途 (關鍵字對應)
    # 邏輯：根據關鍵字決定要篩選什麼車
    if any(k in user_input for k in ["跑山", "運動", "熱血", "快", "殺彎"]):
        filters['keywords'] = ["DRG", "JET", "勁戰", "FORCE", "KRV", "R15", "MT", "GSX", "小阿魯"]
        filters['tag'] = "⛰️ 跑山神車"
    elif any(k in user_input for k in ["買菜", "代步", "便宜", "通勤", "輕"]):
        filters['keywords'] = ["GP", "DUKE", "JOG", "WOO", "NICE", "MANY", "CUXI", "VINO"]
        filters['tag'] = "🛒 買菜代步"
    elif any(k in user_input for k in ["長途", "環島", "休旅", "舒服"]):
        filters['keywords'] = ["SMAX", "FORCE", "MMBCU", "KRV", "NMAX", "PCX"]
        filters['tag'] = "🛣️ 長途休旅"
    elif any(k in user_input for k in ["檔車", "打檔"]):
        filters['style_keyword'] = "檔車"
        filters['tag'] = "🏍️ 帥氣檔車"

    return filters

# --- 5. CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #f1f5f9; }
    
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(120deg, #2563eb, #4f46e5);
        padding: 30px 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin:0; }
    
    /* 卡片樣式 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white; border-radius: 10px; border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); overflow: hidden;
        border-top: 4px solid #3b82f6; 
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-top-color: #f43f5e;
    }

    .card-content { padding: 12px; }
    .moto-title {
        font-weight: 700; font-size: 16px; color: #1e293b; margin: 5px 0; height: 45px;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .price-tag { color: #dc2626; font-weight: 800; font-size: 1.2rem; }
    
    /* 標籤 Pill Styles */
    .tag-box { display: flex; gap: 4px; margin-bottom: 5px; flex-wrap: wrap; }
    .pill { padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; }
    .pill-loc { background-color: #dbeafe; color: #1d4ed8; }
    .pill-ai { background-color: #fce7f3; color: #be185d; } /* AI 推薦標籤 */

    /* 聊天室樣式 */
    .stChatMessage { background-color: white; border-radius: 10px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --- 6. 側邊欄 ---
with st.sidebar:
    st.markdown("### 📍 全域設定")
    all_stores = ["全台分店"] + sorted(list(df['Store'].unique()))
    selected_region = st.selectbox("您的所在位置", all_stores)
    st.info("💡 提示：在「AI 顧問」頁面，您可以直接打字告訴我您的需求，例如：「我在高雄有4萬想買買菜車」。")

# --- 7. 資料預處理 ---
current_df = df.copy()
if selected_region != "全台分店":
    current_df = current_df[current_df['Store'] == selected_region]

# --- 8. 主介面 ---
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">🛵 MotoMatch {selected_region if selected_region != '全台分店' else '全台'}</div>
    <div style="opacity:0.9;">AI 智慧媒合 · 懂車更懂你</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 現場庫存", "💬 AI 購車顧問", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: 傳統列表 (維持不變)
# ==========================================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1: keyword = st.text_input("搜尋車名", placeholder="例如: 勁戰")
    with col2: max_budget = st.number_input("預算上限", value=100000, step=5000)

    filtered_df = current_df.copy()
    if keyword: filtered_df = filtered_df[filtered_df['Model'].str.contains(keyword, case=False)]
    filtered_df = filtered_df[filtered_df['Price'] <= max_budget]

    if filtered_df.empty:
        st.warning("無符合車輛。")
    else:
        st.caption(f"找到 {len(filtered_df)} 台車")
        # 顯示前 12 台
        for i in range(0, min(len(filtered_df), 12), 3):
            cols = st.columns(3)
            batch = filtered_df.iloc[i:i+3]
            for col, (_, row) in zip(cols, batch.iterrows()):
                with col:
                    with st.container(border=True):
                        st.image(row['Image_URL'], use_container_width=True)
                        st.markdown(f"""<div class="card-content">
                            <div class="tag-box"><span class="pill pill-loc">{row["Store"]}</span></div>
                            <div class="moto-title">{row["Model"]}</div>
                            <div class="price-tag">${row["Price"]:,.0f}</div>
                        </div>""", unsafe_allow_html=True)
                        st.link_button("查看", row['Shop_Link'], use_container_width=True)

# ==========================================
# Tab 2: 💬 AI 購車顧問 (核心修改區)
# ==========================================
with tab2:
    st.markdown("### 🤖 MotoBot 智慧助理")
    st.caption("請直接輸入您的需求，例如：**「我在高雄，預算5萬以內，想找一台適合跑山的車」**")

    # 初始化聊天記錄
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。請告訴我您的**地點、預算**以及**用途**（例如：跑山、買菜、長途），我直接幫您找車！"}]

    # 顯示歷史訊息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 處理使用者輸入
    if prompt := st.chat_input("請輸入您的需求..."):
        # 1. 顯示使用者輸入
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. AI 分析 (關鍵字解析)
        # 傳入 all_stores 列表以供比對
        store_list = list(df['Store'].unique())
        intent = parse_user_intent(prompt, store_list)
        
        # 3. 篩選資料
        ai_df = df.copy()
        
        # 條件 A: 地點 (如果使用者有說「高雄」，就只搜高雄；沒說就用側邊欄設定)
        if 'store' in intent:
            ai_df = ai_df[ai_df['Store'] == intent['store']]
            location_msg = f"📍 {intent['store']}"
        elif selected_region != "全台分店":
            ai_df = ai_df[ai_df['Store'] == selected_region]
            location_msg = f"📍 {selected_region}"
        else:
            location_msg = "📍 全台搜尋"

        # 條件 B: 預算
        if 'budget' in intent:
            ai_df = ai_df[ai_df['Price'] <= intent['budget']]
            budget_msg = f"💰 {intent['budget']/10000:.1f}萬內"
        else:
            budget_msg = "💰 預算不限"

        # 條件 C: 用途/車款
        tag_msg = ""
        if 'keywords' in intent:
            # 使用 Regex 模糊比對多個關鍵字
            pattern = '|'.join(intent['keywords'])
            ai_df = ai_df[ai_df['Model'].str.contains(pattern, case=False, regex=True)]
            tag_msg = f"🏷️ {intent['tag']}"
        elif 'style_keyword' in intent:
             ai_df = ai_df[ai_df['Model'].str.contains("檔", na=False)]
             tag_msg = "🏷️ 檔車魂"

        # 4. 產生回應 (不顯示囉嗦的文字，直接給結果)
        result_count = len(ai_df)
        
        with st.chat_message("assistant"):
            if result_count > 0:
                # 簡單的 Summary，不廢話
                st.markdown(f"**分析完畢！條件：{location_msg} 、 {budget_msg} {tag_msg}**")
                st.markdown(f"為您精選以下 **{min(result_count, 3)}** 台最適合的車：")
                
                # 直接顯示卡片 (不存入 session state，避免重複渲染卡頓)
                cols = st.columns(3)
                for i in range(min(result_count, 3)):
                    row = ai_df.iloc[i]
                    with cols[i]:
                        with st.container(border=True):
                            st.image(row['Image_URL'], use_container_width=True)
                            st.markdown(f"""<div class="card-content">
                                <div class="tag-box">
                                    <span class="pill pill-loc">{row["Store"]}</span>
                                    <span class="pill pill-ai">AI 推薦</span>
                                </div>
                                <div class="moto-title">{row["Model"]}</div>
                                <div class="price-tag">${row["Price"]:,.0f}</div>
                            </div>""", unsafe_allow_html=True)
                            st.link_button("👉 查看", row['Shop_Link'], use_container_width=True)
                
                # 為了讓對話延續，我們把「簡短的結論」存入歷史，但卡片不存(太佔空間)
                st.session_state.messages.append({"role": "assistant", "content": f"已為您展示 {location_msg} 預算 {budget_msg} 的推薦車款。還有其他需求嗎？"})
            
            else:
                st.error(f"抱歉，在 {location_msg} 找不到 {budget_msg} 的車款。")
                st.write("建議：試著提高一點預算，或是改搜尋「全台分店」？")
                st.session_state.messages.append({"role": "assistant", "content": "抱歉，找不到符合條件的車，建議調整搜尋條件。"})

# ==========================================
# Tab 3: 🔮 猜你喜歡 (維持功能但美化)
# ==========================================
with tab3:
    if current_df.empty:
        st.error("無資料。")
    else:
        st.info("💡 選一台您喜歡的車，系統會算出「基因最像」的車款。")
        local_sim = build_similarity_model(current_df)
        
        c1, c2 = st.columns(2)
        with c1: ai_brand = st.selectbox("品牌", list(current_df['Brand'].unique()), key="ai_b")
        with c2: ai_target = st.selectbox("車款", current_df[current_df['Brand']==ai_brand]['Model'].unique(), key="ai_t")
            
        if st.button("🚀 啟動關聯推算", type="primary"):
            st.divider()
            try:
                target_idx = current_df.reset_index(drop=True)[current_df.reset_index(drop=True)['Model'] == ai_target].index[0]
                scores = sorted(list(enumerate(local_sim[target_idx])), key=lambda x: x[1], reverse=True)[1:4]
                
                cols = st.columns(3)
                for i, (idx, score) in enumerate(scores):
                    r = current_df.reset_index(drop=True).iloc[idx]
                    with cols[i]:
                        with st.container(border=True):
                            st.image(r['Image_URL'], use_container_width=True)
                            st.caption(f"🧬 相似度 {int(score*100)}%")
                            st.markdown(f"**{r['Model']}**")
                            st.markdown(f'<div class="price-tag">${r["Price"]:,.0f}</div>', unsafe_allow_html=True)
                            st.link_button("查看", r['Shop_Link'], use_container_width=True)
            except:
                st.error("運算失敗")

st.markdown("<br><hr><div style='text-align:center;color:gray'>MotoMatch © 2026</div>", unsafe_allow_html=True)