import streamlit as st
import pandas as pd
import numpy as np
import math
import re

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

# --- 3. 核心演算法 ---
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

# --- 4. CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        padding: 25px 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0, 0.2);
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; margin:0; }
    
    /* 卡片樣式 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white; border-radius: 10px; border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); overflow: hidden;
        border-top: 4px solid #3b82f6; 
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-top-color: #f97316;
    }

    .card-content { padding: 12px; }
    .moto-title {
        font-weight: 700; font-size: 16px; color: #1e293b; margin: 5px 0; height: 45px;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .price-tag { color: #dc2626; font-weight: 800; font-size: 1.3rem; margin-top:5px; }
    
    /* 標籤 */
    .tag-box { display: flex; gap: 5px; margin-bottom: 8px; flex-wrap: wrap; }
    .pill { padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .pill-loc { background-color: #eff6ff; color: #1d4ed8; }
    .pill-abs { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; } 
    .pill-ship { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; } 

    /* 聊天室氣泡 */
    .stChatMessage { background-color: white; border-radius: 10px; border: 1px solid #e2e8f0; }
    
    /* 輸入框固定底部 */
    section[data-testid="stBottomBlock"] {
        background-color: #f8fafc;
        padding-bottom: 20px;
    }
    
    /* 免責聲明文字 */
    .disclaimer-text {
        font-size: 0.8rem; color: #64748b; line-height: 1.5;
    }
    
    /* 分頁按鈕樣式 */
    div.stButton > button {
        width: 100%; border-radius: 8px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 5. 側邊欄 ---
with st.sidebar:
    st.markdown("### 📍 全域設定")
    all_stores = ["全台分店"] + sorted(list(df['Store'].unique()))
    selected_region = st.selectbox("您的所在位置", all_stores)
    
    st.divider()
    
    st.warning("遇到問題嗎？")
    if st.button("🔄 重置 AI 對話", type="primary"):
        st.session_state.chat_stage = 0
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？(例如：高雄)"}]
        st.rerun()

# --- 6. 資料預處理 ---
current_df = df.copy()
if selected_region != "全台分店":
    current_df = current_df[current_df['Store'] == selected_region]

# --- 7. 主介面 ---
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">🛵 MotoMatch {selected_region if selected_region != '全台分店' else '全台'}</div>
    <div style="opacity:0.8; margin-top:5px;">AI 智慧媒合 · 懂車更懂你</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 現場庫存", "💬 AI 購車顧問", "🔮 猜你喜歡"])

# ==========================================
# Tab 1: 現場庫存 (含按鈕式分頁)
# ==========================================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1: keyword = st.text_input("搜尋車名", placeholder="例如: 勁戰")
    with col2: max_budget = st.number_input("預算上限", value=150000, step=5000)

    filtered_df = current_df.copy()
    if keyword: filtered_df = filtered_df[filtered_df['Model'].str.contains(keyword, case=False)]
    filtered_df = filtered_df[filtered_df['Price'] <= max_budget]

    if filtered_df.empty:
        st.warning("無符合車輛。")
    else:
        # --- 分頁計算 ---
        ITEMS_PER_PAGE = 12
        if 'page_number' not in st.session_state: st.session_state.page_number = 1
        total_pages = math.ceil(len(filtered_df) / ITEMS_PER_PAGE)
        if st.session_state.page_number > total_pages: st.session_state.page_number = 1

        # 頂部小資訊
        st.caption(f"共找到 {len(filtered_df)} 台車 | 目前第 {st.session_state.page_number} / {total_pages} 頁")

        # 切割資料
        start_idx = (st.session_state.page_number - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        display_df = filtered_df.iloc[start_idx:end_idx]

        # 顯示網格
        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            batch = display_df.iloc[i:i+3]
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

        # --- ★ 底部按鈕式分頁 (Pagination Bar) ★ ---
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 產生頁碼列表邏輯 (1 2 3 ... 50)
        current = st.session_state.page_number
        if total_pages <= 7:
            page_list = list(range(1, total_pages + 1))
        else:
            if current <= 4:
                page_list = [1, 2, 3, 4, 5, "...", total_pages]
            elif current >= total_pages - 3:
                page_list = [1, "...", total_pages - 4, total_pages - 3, total_pages - 2, total_pages - 1, total_pages]
            else:
                page_list = [1, "...", current - 1, current, current + 1, "...", total_pages]

        # 置中按鈕
        total_cols = len(page_list) + 2
        _, mid, _ = st.columns([2, total_cols, 2]) # 左右留白，中間放按鈕
        
        with mid:
            cols = st.columns(total_cols)
            # 上一頁
            if cols[0].button("◀", disabled=(current == 1), key="prev_page"):
                st.session_state.page_number -= 1
                st.rerun()
            
            # 數字按鈕
            for i, p in enumerate(page_list):
                with cols[i + 1]:
                    if p == "...":
                        st.write("...")
                    else:
                        # 如果是當前頁，用 primary 顏色 (紅色)
                        if st.button(str(p), key=f"page_{p}", type="primary" if p == current else "secondary"):
                            st.session_state.page_number = p
                            st.rerun()
            
            # 下一頁
            if cols[-1].button("▶", disabled=(current == total_pages), key="next_page"):
                st.session_state.page_number += 1
                st.rerun()

# ==========================================
# Tab 2: 💬 AI 購車顧問 (保持最新版)
# ==========================================
with tab2:
    st.markdown("### 🤖 MotoBot 智慧助理")
    
    if "chat_stage" not in st.session_state:
        st.session_state.chat_stage = 0
        st.session_state.chat_data = {} 
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是 MotoBot。(1/5) 請問您**居住在哪個縣市**？(例如：高雄、花蓮)"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    current_placeholder = "請輸入回答..."
    stage = st.session_state.chat_stage
    if stage == 0: current_placeholder = "請輸入您的居住縣市 (例如: 高雄)..."
    elif stage == 1: current_placeholder = "請輸入數字預算 (例如: 50000)..."
    elif stage == 2: current_placeholder = "例如: 跑山、買菜、通勤..."
    elif stage == 3: current_placeholder = "請輸入: 是 / 否..."
    elif stage == 4: current_placeholder = "請輸入: 願意 / 不願意..."

    if prompt := st.chat_input(current_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = ""
        should_rerun = True 
        
        # Q1: 地點
        if stage == 0:
            st.session_state.chat_data['location'] = prompt
            response = f"收到，您在 **{prompt}**。(2/5) 請問您的購車**預算上限**是多少？(例如：5萬)"
            st.session_state.chat_stage = 1

        # Q2: 預算
        elif stage == 1:
            try:
                nums = re.findall(r'\d+', prompt)
                if nums:
                    budget = int(nums[0])
                    if budget < 100: budget *= 10000 
                    st.session_state.chat_data['budget'] = budget
                    response = f"好的，預算 **{budget/10000:.1f}萬** 以內。(3/5) 請問您的**主要用途**是？(例如：跑山、買菜、長途通勤)"
                    st.session_state.chat_stage = 2
                else:
                    response = "不好意思，我沒讀到數字。請輸入數字預算 (例如：50000)"
                    should_rerun = False
            except:
                response = "請輸入有效的數字預算。"
                should_rerun = False

        # Q3: 用途
        elif stage == 2:
            st.session_state.chat_data['usage'] = prompt
            tag = "標準車款"
            if any(k in prompt for k in ["跑山", "運動", "快"]): tag = "⛰️ 跑山"
            elif any(k in prompt for k in ["買菜", "代步", "輕"]): tag = "🛒 代步"
            elif any(k in prompt for k in ["長途", "環島"]): tag = "🛣️ 長途"
            elif any(k in prompt for k in ["檔車"]): tag = "🏍️ 檔車"
            st.session_state.chat_data['tag'] = tag
            response = f"了解 ({tag})。(4/5) 安全性確認：您是否需要配備 **ABS 防鎖死煞車系統**？(請回答：需要/不需要)"
            st.session_state.chat_stage = 3

        # Q4: ABS
        elif stage == 3:
            need_abs = False
            if any(k in prompt for k in ["是", "要", "有", "需要", "yes", "y"]):
                need_abs = True
                abs_msg = "✅ 指定 ABS"
            else:
                abs_msg = "⭕ 無強制 ABS"
            st.session_state.chat_data['abs'] = need_abs
            user_loc = st.session_state.chat_data['location']
            response = f"好的 ({abs_msg})。(5/5) 最後一題：\n\n如果 **{user_loc}** 當地沒有符合的車，我們有些分店在其他縣市。您願意支付約 **$1500 託運費** 將車運過去嗎？(請回答：願意/不願意)"
            st.session_state.chat_stage = 4

        # Q5: 運費 & 搜尋
        elif stage == 4:
            accept_shipping = False
            if any(k in prompt for k in ["願意", "好", "可", "yes", "ok"]):
                accept_shipping = True
            
            st.session_state.chat_data['shipping'] = accept_shipping
            
            final_df = df.copy()
            final_df = final_df[final_df['Price'] <= st.session_state.chat_data['budget']]
            if st.session_state.chat_data['abs']:
                final_df = final_df[final_df['Model'].str.contains("ABS", case=False)]
            usage = st.session_state.chat_data['usage']
            if any(k in usage for k in ["跑山", "運動"]):
                final_df = final_df[final_df['Model'].str.contains("DRG|JET|勁戰|FORCE|KRV|R15", case=False, regex=True)]
            elif any(k in usage for k in ["買菜", "代步"]):
                final_df = final_df[final_df['Model'].str.contains("GP|DUKE|JOG|WOO|NICE|MANY|CUXI", case=False, regex=True)]

            user_loc = st.session_state.chat_data['location']
            if accept_shipping:
                loc_text = "全台搜尋 (含託運)"
            else:
                final_df = final_df[final_df['Store'].str.contains(user_loc, na=False)]
                loc_text = f"僅限 {user_loc}"

            count = len(final_df)
            response = f"""
            🎉 **分析完成！**
            - 📍 **範圍**：{loc_text}
            - 💰 **預算**：{st.session_state.chat_data['budget']/10000}萬內
            - 🛠️ **需求**：{st.session_state.chat_data['tag']} / {"✅ 要ABS" if st.session_state.chat_data['abs'] else "⭕ 不限ABS"}
            
            為您找到 **{count}** 台符合的車款：
            """
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
                if count > 0:
                    cols = st.columns(3)
                    for i in range(min(count, 6)):
                        row = final_df.iloc[i]
                        with cols[i % 3]:
                            with st.container(border=True):
                                st.image(row['Image_URL'], use_container_width=True)
                                tags_html = f'<span class="pill pill-loc">{row["Store"]}</span>'
                                if "ABS" in row['Model']: tags_html += ' <span class="pill pill-abs">ABS</span>'
                                if accept_shipping and user_loc not in row['Store']:
                                    tags_html += ' <span class="pill pill-ship">+$1500運</span>'
                                st.markdown(f"""<div class="card-content">
                                    <div class="tag-box">{tags_html}</div>
                                    <div class="moto-title">{row["Model"]}</div>
                                    <div class="price-tag">${row["Price"]:,.0f}</div>
                                </div>""", unsafe_allow_html=True)
                                st.link_button("👉 查看", row['Shop_Link'], use_container_width=True)
                else:
                    st.error(f"抱歉，在 {loc_text} 找不到符合條件的車。\n建議：\n1. 增加預算\n2. 選擇「願意」接受託運")

            st.session_state.chat_stage = 5 
            should_rerun = False 

        # Q5: 結束
        elif stage == 5:
            st.session_state.chat_stage = 0
            st.session_state.messages = [{"role": "assistant", "content": "🔄 已重置對話。請問您現在**居住在哪個縣市**？"}]
            should_rerun = True

        if stage != 4:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        
        if should_rerun:
            st.rerun()

    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# ==========================================
# Tab 3: 🔮 猜你喜歡
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
            except: st.error("運算失敗")

# ==========================================
# Footer & 免責聲明
# ==========================================
st.divider()

with st.expander("⚖️ 免責聲明與服務條款 (Terms of Service) - 點擊展開"):
    st.markdown("""
    <div class="disclaimer-text">
    1. <b>資訊來源</b>：本平台之車輛資料皆由程式自動抓取自第三方網站，僅供學術研究使用。<br>
    2. <b>準確性聲明</b>：本平台不保證資訊之即時性與正確性。實際車況請以店家現場為主。<br>
    3. <b>交易責任</b>：本平台僅提供資訊媒合服務，不參與實際買賣。任何交易糾紛請直接與車行聯繫。<br>
    4. <b>安全提醒</b>：購買二手車輛強烈建議親自試乘、檢查車況，並簽署正式購車合約。
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size: 0.8rem; margin-top: 10px; margin-bottom: 80px;'>
    MotoMatch AI System © 2026 | Designed by MIS Team<br>
    <span style='font-size: 0.7rem;'>本專題僅供學術交流，非營利目的</span>
</div>
""", unsafe_allow_html=True)