# -----------------------------
# SMART AUTO VISUALIZATION
# -----------------------------

st.subheader("📊 Smart Visualizations")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Correlation Heatmap
if len(numeric_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    fig.colorbar(cax)
    st.pyplot(fig)

# Category Comparison
if len(categorical_cols) > 0:
    cat_col = st.selectbox("Select Category for Comparison", categorical_cols)
    
    st.subheader("📊 Category Comparison")
    grouped = df.groupby(cat_col)[selected_col].mean()
    
    fig, ax = plt.subplots()
    grouped.plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Histogram
st.subheader("📉 Distribution")
fig, ax = plt.subplots()
ax.hist(df[selected_col], bins=20)
st.pyplot(fig)

# Box Plot
st.subheader("📦 Box Plot")
fig, ax = plt.subplots()
ax.boxplot(df[selected_col])
st.pyplot(fig)
