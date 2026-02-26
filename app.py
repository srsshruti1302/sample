# -----------------------------
# SMART AUTO VISUALIZATION
# -----------------------------

st.subheader("📊 Smart Visualizations")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# 1️⃣ If dataset has datetime column → Trend Chart
for col in df.columns:
    if "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
            st.subheader("📈 Time Series Trend")
            fig, ax = plt.subplots()
            ax.plot(df[col], df[selected_col])
            ax.set_xlabel("Date")
            ax.set_ylabel(selected_col)
            st.pyplot(fig)
        except:
            pass

# 2️⃣ Correlation Heatmap (if multiple numeric columns)
if len(numeric_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    fig.colorbar(cax)
    st.pyplot(fig)

# 3️⃣ Categorical vs Numeric
if len(categorical_cols) > 0:
    cat_col = st.selectbox("Select Category for Comparison", categorical_cols)
    
    st.subheader("📊 Category Comparison")
    grouped = df.groupby(cat_col)[selected_col].mean()
    
    fig, ax = plt.subplots()
    grouped.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# 4️⃣ Distribution Chart
st.subheader("📉 Distribution Analysis")
fig, ax = plt.subplots()
ax.hist(df[selected_col], bins=20)
st.pyplot(fig)

# 5️⃣ Box Plot for Outliers
st.subheader("📦 Box Plot (Outlier Detection)")
fig, ax = plt.subplots()
ax.boxplot(df[selected_col])
st.pyplot(fig)
