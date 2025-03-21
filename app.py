import streamlit as st
import pandas as pd
import os
import subprocess
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run the prediction model
def run_test_model(file_path):
    try:
        subprocess.run(["python", "test_model.py", file_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error running model: {e}")
        return False

# Function to segregate data
def segregate_data(df):
    df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"])
    df["year"] = df["first_day_of_month"].dt.year
    df["month"] = df["first_day_of_month"].dt.month
    df["week"] = df["first_day_of_month"].dt.isocalendar().week
    return df

def main():
    st.title("Microbusiness Density Prediction & Analysis Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])
    file_path = "data/dummy_test_reduced.csv"  # Default file

    if uploaded_file is not None:
        file_path = "data/uploaded_test.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Running predictions..."):
            success = run_test_model(file_path)
        if success:
            st.success("Predictions completed and saved to `data/submission.csv`." )
    
    if os.path.exists("data/submission.csv"):
        df = pd.read_csv("data/submission.csv", parse_dates=["first_day_of_month"])
        st.table(df.head(15))
        st.subheader("Stacked Bar Chart of Microbusiness Density")

        # Convert first_day_of_month to only date (removing time)
        df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"]).dt.date

        # Pivot DataFrame for stacked bar chart
        pivot_df = df.pivot(index="first_day_of_month", columns="cfips", values="microbusiness_density")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot_df.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)

        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Microbusiness Density")
        ax.set_title("Stacked Bar Chart of Microbusiness Density")

        # Format x-axis to show only date (without time)
        ax.set_xticks(range(0, len(pivot_df.index), max(1, len(pivot_df.index) // 10)))  # Reduce number of x-axis labels
        # ax.set_xticklabels(pivot_df.index.strftime('%Y-%m-%d'), rotation=45)

        # Move legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="CFIPS")

        # Adjust layout to prevent cutting off the legend
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

        st.subheader("Time Series of Microbusiness Density")

        plt.figure(figsize=(12, 6))
        for cfips in df["cfips"].unique():
            subset = df[df["cfips"] == cfips]
            plt.plot(subset["first_day_of_month"], subset["microbusiness_density"], label=f"CFIPS {cfips}")

        plt.xlabel("Date")
        plt.ylabel("Microbusiness Density")
        plt.title("Microbusiness Density Over Time")
        plt.legend()

        # Display in Streamlit
        st.pyplot(plt)

        selected_cfips = st.sidebar.selectbox("Select CFIPS", df["cfips"].unique())

        st.subheader(f"Microbusiness Density for CFIPS {selected_cfips}")

        subset = df[df["cfips"] == selected_cfips]

        plt.figure(figsize=(12, 6))
        plt.plot(subset["first_day_of_month"], subset["microbusiness_density"], label=f"CFIPS {selected_cfips}")

        plt.xlabel("Date")
        plt.ylabel("Microbusiness Density")
        plt.title(f"Microbusiness Density Over Time for CFIPS {selected_cfips}")
        plt.legend()

        st.pyplot(plt)

if __name__ == "__main__":
    main()
