import streamlit as st
from ML_Package import DataLoader, ModelTrainer
from ML_Package import summarize_data, handle_missing_values, encode_categorical_variables, scale_numerical_features

# Initialize the DataLoader class
data_loader = DataLoader()

def streamlit_app():
    st.title("Own PyCaret: AutoML & EDA")

    # Step 1: Upload File
    uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx", "json", "hdf5"])

    if uploaded_file is not None:
        # get file extension to determine file format
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Step 2: Read Data
        data = data_loader.read_data(uploaded_file, file_format=file_extension)
        
        if data is not None:
            if 'data' not in st.session_state:
                st.session_state['data'] = data.copy()
        else:
            st.write("Error: Failed to load data.")

        # Step 3: Print Data 
        if data is not None:
            st.write("### Data Preview:")
            st.write(st.session_state['data'].head())

            # Step 4: Target Vraiable Selection
            target_variable = st.selectbox(
                "Select the target variable for prediction:",
                options=["Select a target variable..."] + list(st.session_state['data'].columns)
            )

        
            if target_variable != "Select a target variable...":
                st.write(f"Target variable selected: **{target_variable}**")

                st.subheader("Exploratory Data Analysis (EDA)")

            # Step 5: EDA
                # Summary Statistics
                if st.checkbox("Run Summary Statistics"):
                    st.write("**Summary Statistics**")
                    st.write(summarize_data(st.session_state['data']))

                # Handle Missing Values
                if st.checkbox("Handle Missing Values"):
                    num_strategy = st.selectbox("Numerical Strategy", options=["mean", "median", "most_frequent", "remove"])
                    cat_strategy = st.selectbox("Categorical Strategy", options=["most_frequent", "remove"])
                    if st.button("Apply Missing Values Strategy"):
                        st.session_state['data'] = handle_missing_values(st.session_state['data'], num_strategy=num_strategy, cat_strategy=cat_strategy)
                        st.write("Missing values handled successfully.")
                        st.write(st.session_state['data'].head()) 

                # Encoding Categorical Variables
                if st.checkbox("Encode Categorical Variables"):
                    if st.button("Encode"):
                        st.session_state['data'] = encode_categorical_variables(st.session_state['data'])
                        st.write("Categorical variables encoded.")
                        st.write(st.session_state['data'].head()) 

                # Scaling Numerical Features
                if st.checkbox("Scale Numerical Features"):
                    if st.button("Scale"):
                        st.session_state['data'] = scale_numerical_features(st.session_state['data'], target_variable)
                        st.write("Numerical features scaled (excluding target variable).")
                        st.write(st.session_state['data'].head()) 

                # finalize the EDA section and move to training
                if st.button("Done with EDA"):
                    st.session_state['eda_done'] = True 

            # Step 6: Model Training and Evaluation
            if 'eda_done' in st.session_state and st.session_state['eda_done']:
                st.subheader("Model Training and Selection")

                # Automatically determine task type based on the target variable
                if st.session_state['data'][target_variable].dtype == 'object' or st.session_state['data'][target_variable].nunique() <= 10:
                    task_type = 'classification'
                else:
                    task_type = 'regression'

                st.write(f"Detected task type: **{task_type}**")

                # Initialize ModelTrainer
                model_trainer = ModelTrainer(st.session_state['data'], target_variable, task_type)

                # Option 1: Use AutoML with PyCaret
                if st.checkbox("Run AutoML"):
                    best_model = model_trainer.auto_train_and_evaluate()
                    st.session_state['best_model'] = best_model  # Save the best AutoML model in session state
                    st.session_state['model_source'] = "AutoML"  # Keep track of where the model came from
                    st.write(f"Best Model from AutoML: {best_model}")
                
                # Option 2: Manual Model Selection
                model_choices = list(model_trainer.get_models().keys())
                selected_model = st.selectbox("Choose a model to train manually:", model_choices)

                if st.checkbox(f"Train {selected_model}"):
                    manual_model, score = model_trainer.train_and_evaluate(selected_model)
                    st.session_state['best_model'] = manual_model
                    st.session_state['model_source'] = "Manual"
                    st.session_state['model_score'] = score
                    st.write(f"Trained {selected_model} with Score: {score}")

                # Option to tune the best model from AutoML
                if 'best_model' in st.session_state:
                    st.write(f"Best model source: {st.session_state['model_source']}")

                    if st.checkbox("Tune Best AutoML Model"):
                        tuned_model = model_trainer.tune_best_model()
                        st.session_state['best_model'] = tuned_model
                        st.write(f"Tuned Model: {tuned_model}")
                else:
                    st.write("Train a model using AutoML or manual selection before tuning.")

# Run the Streamlit app
if __name__ == '__main__':
    streamlit_app()
