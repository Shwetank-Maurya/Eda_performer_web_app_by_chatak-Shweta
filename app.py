import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="📊 EDA Performer Web App by Chatak-Shweta", layout="wide")
st.title("📈 EDA Performer Web App By Chatak-Shweta")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def basic_info(df):
    st.subheader("📊 Basic Info")
    st.write(f"Shape: {df.shape}")
    st.write(f"Number of elements :{df.size}")
    st.write("Data Types:")
    st.write(df.dtypes)
    st.subheader("🧾 Descriptive Statistics")
    st.write(df.describe(include='all'))

def extreme_values_table(df):
    st.subheader("⚠️ Extreme Values")
    numeric_cols = df.select_dtypes(include='number').columns
    data = {'Column': [], 'Min': [], 'Max': [], 'Min Index': [], 'Max Index': []}
    for col in numeric_cols:
        data['Column'].append(col)
        data['Min'].append(df[col].min())
        data['Max'].append(df[col].max())
        data['Min Index'].append(df[col].idxmin())
        data['Max Index'].append(df[col].idxmax())
    extreme_df = pd.DataFrame(data)
    st.dataframe(extreme_df)

def missing_values(df):
    st.subheader("📉 Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write(missing)
    else:
        st.write("No missing values.")

def user_input_column(df):
    st.subheader("🔍 Explore Individual Columns")
    col = st.selectbox("Select a column", options=['None'] + df.columns.tolist())

    if col == 'None':
        st.info("Please select a column to display details.")
        return

    st.write(f"📌 **Column Selected:** `{col}`")
    st.write(f"🧠 Data Type: `{df[col].dtype}`")
    st.write(f"🔢 Unique Values: {df[col].nunique()}")
    st.write(f"📈 Distinct (%): {(df[col].nunique() / df[col].count()) * 100:.2f}%")
    st.write(f"❓ Missing Values: {df[col].isnull().sum()}")
    st.write(f"❗ Missing (%): {(df[col].isnull().sum() / df[col].count()) * 100:.2f}%")

    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(f"📏 Mean: {df[col].mean():.2f}")

    
    col1, col2= st.columns(2)
    with col1:
        show_stats = st.button("📊 Show Statistics", key=f"stats_{col}")
    with col2:
        show_hist = st.button("📈 Show Histogram", key=f"hist_{col}")

    if show_stats:
        st.write(df[col].describe())

    if show_hist and pd.api.types.is_numeric_dtype(df[col]):
        fig, ax = plt.subplots(figsize=(4,3))
        df[col].dropna().hist(ax=ax,bins=10, color='lightgreen', edgecolor='black')
        ax.set_title(f"Histogram of {col}",fontsize=10,pad=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        
        
        st.pyplot(fig, use_container_width=False)
        
    elif show_hist:
        st.warning("Histogram only available for numeric columns.")

def visualize(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns to visualize.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_hist = st.button("📊 Show Histograms")
    with col2:
        show_pairplot = st.button("📈 Show Pair Plot")
    with col3:
        show_correlation = st.button("🔗 Show Correlation Matrix")
    with col4:
        show_pieplot = st.button("♾️ Show Pie Chart")

    if show_hist:
        st.subheader("📈 Histograms")
        cols_per_row = 3
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(numeric_cols): 
            with cols[i % cols_per_row]:
                fig, ax = plt.subplots()
                df[col].dropna().hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
                plt.close(fig)

    if show_pairplot:
        st.subheader("📊 Pair Plot")
        try:
            with st.spinner("Generating pair plot..."):
                
                fig = sns.pairplot(df[numeric_cols], 
                                kind="scatter", 
                                plot_kws={'color': "#1F1FC0", 'alpha': 0.6},
                                diag_kws={'color': "#df1481"})
                
                
                fig.fig.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.error(f"Could not generate pair plot: {e}")

    if show_correlation:
        st.subheader("🔗 Correlation Heatmap")
        with st.spinner("Calculating correlations..."):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm',linecolor='black',square=True,linewidths=0.5, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    if show_pieplot:
        st.subheader("📈 Pie Plot")
        cols_per_row = 3
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(numeric_cols):
            with cols[i % cols_per_row]:
                fig, ax = plt.subplots()
                try:
                    binned = pd.cut(df[col], bins=5)
                    counts = binned.value_counts().sort_index()
                    ax.pie(
                        counts.values,
                        labels=[str(interval) for interval in counts.index],
                        autopct='%1.1f%%',
                        startangle=0
                    )
                    ax.set_title(f"Pie Chart of {col}")
                    ax.axis('equal')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not create pie chart for {col}: {str(e)}")
                plt.close(fig)

def plot_numerical(feature, df):
    if not pd.api.types.is_numeric_dtype(df[feature]):
        st.warning(f"Cannot plot {feature} - not a numerical column")
        return
        
    plt.figure(figsize=(29,10))
    plt.subplot(1,2,1)
    sns.histplot(data=df, x=feature, kde=True)
    plt.xlabel(feature, fontsize=12, weight="bold")
    plt.ylabel("Freq", fontsize=12, weight="bold")
    plt.subplot(1,2,2)
    sns.boxplot(data=df, x=feature)
    plt.xlabel(feature, fontsize=12, weight="bold")
    plt.suptitle(feature, weight="bold", fontsize=20, color='blue')
    st.pyplot(plt)
    plt.close()

def skewness(df):
    for feature in df.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue  
            
        col1, col2 = st.columns(2)
        with col1:
            skew_value = (sum((df[feature] - np.mean(df[feature])) ** 3) / len(df[feature])) / (np.std(df[feature]) ** 3)
            st.write(f"Skewness of {feature}: {skew_value}")
            if skew_value > 0.5:
                st.write("Right Skewed")
            elif -0.5 <= skew_value <= 0.5:
                st.write("Not Skewed")
            else:
                st.write("Left Skewed")
        with col2:
            plot_numerical(feature, df)
def print(df):
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_full = st.button("Full DataFrame View")
    with col2:
        show_head = st.button("Head DataFrame View")
    with col3:
        show_tail = st.button("Tail DataFrame View")
    with col4:
        show_random=st.button("Random DataFrame View")
    if show_full:
        st.write(df)
    if show_head:
        st.write(df.head())
    if show_tail:
        st.write(df.tail())
    if show_random:
        st.write(df.sample(5))

    
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        tab1, tab2, tab3, tab4,tab5,tab6= st.tabs([
            "📊 Basic Info", 
            "⚠️ Extreme Values", 
            "🔍 Column Explorer", 
            "📈 Visualizations",
            "📊 Skewness ",
            "🔗 Data overview"
        ])
        
        with tab1:
            basic_info(df)
        
        with tab2:
            extreme_values_table(df)
            missing_values(df)
        
        with tab3:
            user_input_column(df)
        
        with tab4:
            visualize(df)

        with tab5:
            skewness(df)
    
        with tab6:
            print(df)
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("📂 Please upload a CSV file to begin analysis.")