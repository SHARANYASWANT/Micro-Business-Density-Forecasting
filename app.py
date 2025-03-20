import streamlit as st
import pandas as pd
import os
import subprocess
import state_county_mappings

# Dummy mapping dictionaries. Replace these with actual mappings.
state_to_cfips=state_county_mappings.state_to_cfips 
county_to_cfips=state_county_mappings.county_to_cfips

def run_test_model(file_path="data/dummy_test_2026.csv"):
    """
    Runs the backend model script with the given file path.
    """
    try:
        subprocess.run(["python", "test_model.py", file_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error running model: {e}")
        return False

def segregate_data(df):
    """
    Adds datetime features and prepares for filtering.
    """
    df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"])
    df["year"] = df["first_day_of_month"].dt.year
    df["month"] = df["first_day_of_month"].dt.month
    df["week"] = df["first_day_of_month"].dt.isocalendar().week
    return df
def main():
    st.title("Microbusiness Density Prediction & Analysis Dashboard")

    # File uploader
    uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])
    predictions_made = True
    file_path = "data/dummy_test_2026.csv"  # Default file

    # If a file is uploaded, save it
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
            st.success("Predictions completed and saved to `data/submission.csv`.")
            predictions_made = True

    # Load and display predictions only if predictions were made
    if predictions_made and os.path.exists("data/submission.csv"):
        st.subheader("Prediction Results")
        df = pd.read_csv("data/submission.csv")
        st.write(df.head())

        # Load train.csv to join with `cfips` metadata
        df_train = pd.read_csv("data/train.csv")
        df = df.merge(df_train[["cfips", "first_day_of_month"]], on="cfips", how="left")
        df = segregate_data(df)

        # General Analytics Graphs
        st.subheader("General Analytics")
        example_cfips = df["cfips"].unique()[0]
        example_df = df[df["cfips"] == example_cfips].sort_values("first_day_of_month")
        st.line_chart(example_df.set_index("first_day_of_month")["microbusiness_density"])

        # Segregation Options
        st.subheader("Segregated Analysis")
        analysis_type = st.selectbox("Select Analysis Type", 
                                     ["Monthly Average", "Yearly Average", "Weekly (State)", "County"])
        
        if analysis_type == "Monthly Average":
            monthly_avg = df.groupby(["year", "month"])["microbusiness_density"].mean().reset_index()
            st.write("Monthly Average:")
            st.dataframe(monthly_avg)
            monthly_avg["year_month"] = monthly_avg["year"].astype(str) + "-" + monthly_avg["month"].astype(str)
            st.line_chart(monthly_avg.set_index("year_month")["microbusiness_density"])
        
        elif analysis_type == "Yearly Average":
            yearly_avg = df.groupby("year")["microbusiness_density"].mean().reset_index()
            st.write("Yearly Average:")
            st.dataframe(yearly_avg)
            st.bar_chart(yearly_avg.set_index("year"))
        
        elif analysis_type == "Weekly (State)":
            state_name = st.text_input("Enter State Name (e.g., California)", "California")
            cfips_list = state_to_cfips.get(state_name)
            if cfips_list is None:
                st.error("State not found in mapping. Please check your input.")
            else:
                state_df = df[df["cfips"].isin(cfips_list)]
                weekly_avg = state_df.groupby(["year", "week"])["microbusiness_density"].mean().reset_index()
                st.write(f"Weekly Average for {state_name}:")
                st.dataframe(weekly_avg)
                weekly_avg["year_week"] = weekly_avg["year"].astype(str) + "-" + weekly_avg["week"].astype(str)
                st.line_chart(weekly_avg.set_index("year_week")["microbusiness_density"])
        
        elif analysis_type == "County":
            county_name = st.text_input("Enter County Name (e.g., Los Angeles County)", "Los Angeles County")
            cfips_val = county_to_cfips.get(county_name)
            if cfips_val is None:
                st.error("County not found in mapping. Please check your input.")
            else:
                county_df = df[df["cfips"] == cfips_val]
                county_avg = county_df.groupby(["year", "month"])["microbusiness_density"].mean().reset_index()
                st.write(f"Monthly Average for {county_name}:")
                st.dataframe(county_avg)
                county_avg["year_month"] = county_avg["year"].astype(str) + "-" + county_avg["month"].astype(str)
                st.line_chart(county_avg.set_index("year_month")["microbusiness_density"])

if __name__ == "__main__":
    main()
