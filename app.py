import streamlit as st
import pandas as pd
import numpy as np

# --- 1. 網頁設定 ---
st.set_page_config(page_title="MotoMatch 完整版", page_icon="🛵", layout="wide")

# --- 2. 讀取資料 ---
@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("labeled_data.csv")
    except FileNotFoundError:
        return pd.DataFrame() 
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Image_URL'] = df['Image_URL'].fillna('https://cdn-icons-png.flaticon.com/512/3097/3097180.png')
    if 'Store' not in df.columns: df['Store'] = '全台分店'
    if 'Brand' not in df.columns: df['Brand'] = '其他'
    if 'Style' not in df.columns: df['Style'] = '通勤'
    
    # 建立唯一 ID
    df['id'] = df.index
    return df

df = load_data()

# --- 3. 核心演算法：手刻餘弦相似度 (免安裝 sklearn) ---
@st.cache_resource
def build_similarity_model(data):
    # A. 價格正規化
    max_price = data['Price'].max()
    if max_price == 0: max_price = 1
    price_norm = data[['Price']] / max_price
    
    # B. 獨熱編碼 (加權)
    brands_ohe = pd.get_dummies(data['Brand']) * 1.5 
    styles_ohe = pd.get_dummies(data['Style']) * 1.5 
    
    # C. 組合特徵
    features = np.hstack([price_norm.values, brands_ohe.values, styles_ohe.values])
    
    # D. 餘弦相似度公式
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    features_normalized = features / norm
    cosine_sim = np.dot(features_normalized, features_normalized.T)
    
    return cosine_sim

if not df.empty:
    similarity_matrix = build_similarity_model(df)

# --- 4. CSS 美化 ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem;}
    
    /* 區塊標題 */
    .section-title {
        font-size: 24px; font-weight: bold; color: #1e3a8a; 
        border-left: 5px solid #3b82f6; padding-left: 10px; margin-top: 20px; margin-bottom: 20px;
    }
    
    /* 卡片樣式 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #f0f0f0; border-radius: 12px; transition: 0.3s;
        background-color: white; padding: 0 !important; overflow: hidden;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #3b82f6; transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* 卡片內文字 */
    .card-content { padding: 12px; }
    .moto-title { font-weight: 700; font-size: 16px; margin-bottom: 5px; height: 44px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .moto-price { font-size: 18px; font-weight: 800; color: #ef4444; }
    .store-tag { font-size: 12px; color: #6b7280; background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
    
    /* AI 推薦標籤 */
    .rec-tag { background-color: #8b5cf6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-bottom: 5px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# --- 5. 側邊欄：雙重控制 ---
with st.sidebar:
    st.title("🛵 MotoMatch")
    
    # --- Part 1: 上半部篩選器 ---
    st.header("🔍 列表篩選條件")
    filter_keyword = st.text_input("搜尋車名 (例如: Jet, 勁戰)")
    filter_brand = st.multiselect("品牌", sorted(df['Brand'].unique()), default=[])
    filter_budget = st.slider("預算上限", 0, 150000, 150000, step=5000)
    
    st.divider()
    
    # --- Part 2: 下半部 AI 設定 ---
    st.header("🔮 AI 推算設定")
    st.info("此設定控制下方的「猜你喜歡」區塊")
    # 為了讓 AI 選擇器不要太長，先選品牌
    ai_brand_filter = st.selectbox("AI 種子車品牌", ["全部"] + list(df['Brand'].unique()))
    
    if ai_brand_filter != "全部":
        ai_options = df[df['Brand'] == ai_brand_filter]
    else:
        ai_options = df
        
    ai_selected_car = st.selectbox("選擇一台基準車", ai_options['Model'].unique())

# ==========================================
# 🛑 第一部分：全台車庫總覽 (列表顯示)
# ==========================================
st.markdown('<div class="section-title">🏆 全台車庫總覽</div>', unsafe_allow_html=True)

# 1. 執行篩選
list_df = df.copy()
if filter_keyword:
    list_df = list_df[list_df['Model'].str.contains(filter_keyword, case=False, na=False)]
if filter_brand:
    list_df = list_df[list_df['Brand'].isin(filter_brand)]
list_df = list_df[list_df['Price'] <= filter_budget]

# 2. 分頁邏輯
if 'page_number' not in st.session_state: st.session_state.page_number = 1
ITEMS_PER_PAGE = 8 # 上半部顯示少一點，讓這頁不要太長
total_pages = max(1, -(-len(list_df) // ITEMS_PER_PAGE)) # Ceiling division

col_pg1, col_pg2 = st.columns([6, 2])
with col_pg1: st.caption(f"共找到 {len(list_df)} 台車")
with col_pg2: current_page = st.number_input("頁數", 1, total_pages, key="page_input")

start_idx = (current_page - 1) * ITEMS_PER_PAGE
display_df = list_df.iloc[start_idx : start_idx + ITEMS_PER_PAGE]

# 3. 顯示網格
if display_df.empty:
    st.warning("沒有符合條件的車輛。")
else:
    for i in range(0, len(display_df), 4): # 一行 4 個
        cols = st.columns(4)
        for col, (_, row) in zip(cols, display_df.iloc[i:i+4].iterrows()):
            with col:
                with st.container(border=True):
                    try: st.image(row['Image_URL'], use_container_width=True)
                    except: st.empty()
                    
                    st.markdown('<div class="card-content">', unsafe_allow_html=True)
                    st.markdown(f'<span class="store-tag">📍 {row["Store"]}</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="moto-title" title="{row["Model"]}">{row["Model"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="moto-price">${row["Price"]:,.0f}</div>', unsafe_allow_html=True)
                    st.link_button("查看詳情", row['Shop_Link'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔮 第二部分：AI 關聯推算 (餘弦相似度)
# ==========================================
st.markdown("---") # 分隔線
st.markdown('<div class="section-title">猜你喜歡 </div>', unsafe_allow_html=True)
st.markdown("不用搜尋！系統根據您在左下角選擇的 **基準車輛**，自動計算基因最像的車款。")

if df.empty:
    st.error("資料庫為空，無法執行 AI 運算。")
else:
    # 找出使用者在側邊欄選的那台車
    target_car = df[df['Model'] == ai_selected_car].iloc[0]
    
    # 顯示種子車 (左邊) 與 推薦結果 (右邊)
    col_seed, col_recs = st.columns([1, 3])
    
    with col_seed:
        st.info("🎯 您的基準車")
        with st.container(border=True):
            try: st.image(target_car['Image_URL'], use_container_width=True)
            except: st.empty()
            st.markdown('<div class="card-content">', unsafe_allow_html=True)
            st.markdown(f"**{target_car['Model']}**")
            st.caption(f"💰 ${target_car['Price']:,.0f} | {target_car['Brand']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col_recs:
        st.success("🧬 演算法推算結果")
        
        # 執行推算
        try:
            # 取得該車的相似度向量
            sim_scores = list(enumerate(similarity_matrix[target_car['id']]))
            # 排序 (排除自己)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
            
            rec_cols = st.columns(3)
            for i, (idx, score) in enumerate(sim_scores):
                rec_car = df.iloc[idx]
                with rec_cols[i]:
                    with st.container(border=True):
                        # 圖片
                        try: st.image(rec_car['Image_URL'], use_container_width=True)
                        except: st.empty()
                        
                        st.markdown('<div class="card-content">', unsafe_allow_html=True)
                        # 相似度標籤
                        st.markdown(f'<div class="rec-tag">🧬 相似度 {int(score*100)}%</div>', unsafe_allow_html=True)
                        
                        st.markdown(f'<div class="moto-title">{rec_car["Model"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="moto-price">${rec_car["Price"]:,.0f}</div>', unsafe_allow_html=True)
                        
                        # 解釋原因
                        reasons = []
                        if rec_car['Brand'] == target_car['Brand']: reasons.append("同品牌")
                        if rec_car['Style'] == target_car['Style']: reasons.append("同風格")
                        if abs(rec_car['Price'] - target_car['Price']) < 5000: reasons.append("價格接近")
                        st.caption(f"💡 {'、'.join(reasons)}")
                        
                        st.link_button("查看", rec_car['Shop_Link'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"運算錯誤: {e}")

# --- 頁尾 ---# --- 頁尾免責聲明 ---
st.markdown("---")
with st.expander("⚖️ 免責聲明與服務條款 (Terms of Service)"):
    st.markdown("""
    1. **資訊來源**：本平台車輛資料皆自動抓取自第三方網站（貳輪嶼），本平台不保證資訊之即時性、正確性或完整性。
    2. **交易責任**：本平台僅提供資訊媒合與推薦服務，不參與實際買賣、過戶或金流。所有交易糾紛請直接與車行聯繫。
    3. **車況擔保**：二手車況千變萬化，強烈建議買家務必親自前往門市試乘、檢查，並簽署正式購車合約。
    4. **下架機制**：系統會定期更新資料，但若遇車輛已售出未即時下架，請以店家現場庫存為主。
    """)
    
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    MotoMatch © 2026 | 
    <a href='https://shop.2motor.tw/' target='_blank'>資料來源：貳輪嶼車業</a> | 
    專題製作：資管系開發團隊
</div>
""", unsafe_allow_html=True)
st.markdown("<br><hr><div style='text-align:center; color:gray;'>MotoMatch AI System © 2026</div>", unsafe_allow_html=True)