
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import time
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import database as db

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Data Analysis Dashboard")
st.write("Upload a CSV file to analyze, visualize, and gain insights from your data without writing complex code.")

# Important variables
categorical_columns = []  # Store the Categorical Column Name
numerical_columns = []    # Store the Numerical Column Name
missing_null_values = []  # storing the missing values
user_data_frame = None    # User data frame after modifications

# Upload a File
uploaded_file = st.file_uploader("Upload a CSV File", type='CSV')
if uploaded_file:
    with st.spinner('Loading File...'):
        time.sleep(1)  # Reduced time for better UX
    st.success(f'Successfully Uploaded File: {uploaded_file.name}')

if uploaded_file is not None:
    # Add error handling for CSV parsing
    try:
        data_frame = pd.read_csv(uploaded_file)
        user_data_frame = data_frame.copy()
        
        # Categorize columns as categorical or numerical
        categorical_columns = []
        numerical_columns = []
        for col in data_frame.columns:
            if data_frame[col].dtype == "object":
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Data Overview", 
            "Missing Values", 
            "Data Filtering", 
            "Visualization", 
            "Correlation Analysis",
            "Advanced Analysis",
            "Database"
        ])
        
        with tab1:
            st.header("Data Overview")
            
            # Show dataframe with pagination
            st.subheader(f'{uploaded_file.name}: DataFrame')
            st.dataframe(data_frame)
            
            # Display summary statistics
            st.subheader("Statistical Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Rows", data_frame.shape[0])
                st.metric("Number of Columns", data_frame.shape[1])
                st.metric("Numerical Columns", len(numerical_columns))
                st.metric("Categorical Columns", len(categorical_columns))
            
            with col2:
                total_missing = data_frame.isnull().sum().sum()
                st.metric("Missing Values", total_missing)
                st.metric("Memory Usage", f"{data_frame.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            # Statistical Information
            statistical_options = ['Column Names', 'Statistics Data', 'Data Types', 'Null Values']
            statistical_user_data = st.multiselect('View Basic Information', options=statistical_options)
            
            if 'Column Names' in statistical_user_data:
                columns = data_frame.columns
                st.subheader(f'Columns in {uploaded_file.name} File:')
                st.write(columns)
                
            if 'Statistics Data' in statistical_user_data:
                description = data_frame.describe(include='all')
                st.subheader(f'Statistics in {uploaded_file.name} File:')
                st.write(description)
                
            if 'Data Types' in statistical_user_data:
                data_types = data_frame.dtypes
                st.subheader(f'Data Types in {uploaded_file.name} File:')
                st.write(data_types)
                
            if 'Null Values' in statistical_user_data:
                missing_null_values = data_frame.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_null_values.index,
                    'Missing Values': missing_null_values.values,
                    'Percentage': (missing_null_values.values / len(data_frame) * 100).round(2)
                })
                st.subheader(f'Null Values in {uploaded_file.name} File:')
                st.dataframe(missing_df)
                
                # Visualize missing values
                if missing_null_values.sum() > 0:
                    fig = px.bar(
                        missing_df[missing_df['Missing Values'] > 0], 
                        x='Column', 
                        y='Missing Values',
                        text='Percentage',
                        title='Missing Values Distribution'
                    )
                    fig.update_traces(texttemplate='%{text}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Finding unique values in categorical columns
            st.subheader('Unique Values in Categorical Columns')
            unique_column_options = categorical_columns.copy()
            unique_column_options.append('Entire DataFrame')
            
            unique_column_user = st.multiselect('Select columns to view unique values:', options=unique_column_options)
            
            if 'Entire DataFrame' in unique_column_user:
                entire_data_frame = []
                for col in categorical_columns:
                    unique_vals = data_frame[col].unique()
                    unique_counts = data_frame[col].value_counts()
                    
                    entire_data_frame.append({
                        'Column': col,
                        'Unique Values': len(unique_vals),
                        'Most Common': unique_counts.index[0] if len(unique_counts) > 0 else None,
                        'Most Common Count': unique_counts.iloc[0] if len(unique_counts) > 0 else 0
                    })
                
                st.subheader(f'Unique Value Summary for All Categorical Columns:')
                st.dataframe(pd.DataFrame(entire_data_frame))
                unique_column_user.remove('Entire DataFrame')
            
            for col in unique_column_user:
                st.write(f'Unique Values in Column: {col}')
                unique_vals = data_frame[col].value_counts().reset_index()
                unique_vals.columns = [col, 'Count']
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.dataframe(unique_vals)
                    st.write(f'Total Unique Values: {data_frame[col].nunique()}')
                
                with col2:
                    if data_frame[col].nunique() <= 30:  # Only plot if there aren't too many unique values
                        fig = px.pie(unique_vals, names=col, values='Count', title=f'Distribution of {col}')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Too many unique values ({data_frame[col].nunique()}) to display in a pie chart.")
                        # Show top 10 instead
                        top_10 = unique_vals.head(10)
                        fig = px.bar(top_10, x=col, y='Count', title=f'Top 10 values for {col}')
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Handle Missing Values")
            
            if missing_null_values is not None:
                missing_user_data = st.selectbox('Select data type to handle:', options=['Categorical Data', 'Numerical Data'])
                missing_values_info = []
                
                if missing_user_data == 'Categorical Data':
                    st.subheader('Missing Value Columns in Categorical Data')
                    for col in categorical_columns:
                        null_count = data_frame[col].isnull().sum()
                        if null_count > 0:
                            percentage = (null_count / data_frame.shape[0]) * 100
                            missing_values_info.append({'Column': col, 'Count': null_count, 'Percentage': percentage})
                    
                    if missing_values_info:
                        st.table(pd.DataFrame(missing_values_info))
                        choose_col = st.selectbox('Choose a Column:', options=[info['Column'] for info in missing_values_info])
                        
                        if choose_col:
                            method = st.selectbox("Choose a Method:", options=['Most Frequent', 'Constant', 'Drop the Values'])
                            
                            if method == 'Most Frequent':
                                data = user_data_frame[choose_col].fillna(user_data_frame[choose_col].mode()[0])
                                with st.expander('See Preview After Applying Most Frequent Value'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update the Values in Data Frame')
                                if checkbox:
                                    user_data_frame[choose_col] = user_data_frame[choose_col].fillna(user_data_frame[choose_col].mode()[0])
                                    st.success(f"Updated column '{choose_col}' with most frequent value")
                            
                            elif method == 'Constant':
                                const_value = st.text_input('Enter the Constant Value:')
                                if const_value:
                                    data = user_data_frame[choose_col].fillna(const_value)
                                    with st.expander('See Preview After Applying Constant Value'):
                                        st.dataframe(data)
                                    
                                    checkbox = st.checkbox('Update the Values in Data Frame')
                                    if checkbox:
                                        user_data_frame[choose_col] = user_data_frame[choose_col].fillna(const_value)
                                        st.success(f"Updated column '{choose_col}' with constant value: '{const_value}'")
                            
                            elif method == 'Drop the Values':
                                drop_option = st.radio("Choose drop method:", ["Drop only rows with missing values", "Drop entire column"])
                                
                                if drop_option == "Drop only rows with missing values":
                                    data = user_data_frame.dropna(subset=[choose_col])
                                    with st.expander('See Preview After Dropping Rows'):
                                        st.dataframe(data)
                                    
                                    checkbox = st.checkbox('Update Data Frame - Drop Rows')
                                    if checkbox:
                                        user_data_frame = user_data_frame.dropna(subset=[choose_col])
                                        st.success(f"Dropped rows with missing values in column '{choose_col}'")
                                
                                else:  # Drop entire column
                                    data = user_data_frame.drop(columns=[choose_col])
                                    with st.expander('See Preview After Dropping Column'):
                                        st.dataframe(data)
                                    
                                    checkbox = st.checkbox('Update Data Frame - Drop Column')
                                    if checkbox:
                                        user_data_frame = user_data_frame.drop(columns=[choose_col])
                                        categorical_columns.remove(choose_col)
                                        st.success(f"Dropped column '{choose_col}'")
                            
                            with st.expander('Display Updated Data Frame'):
                                st.dataframe(user_data_frame)
                    else:
                        st.success('No Missing Values in Categorical Data')
                
                elif missing_user_data == 'Numerical Data':
                    st.subheader('Missing Value Columns in Numerical Data')
                    for col in numerical_columns:
                        null_count = data_frame[col].isnull().sum()
                        if null_count > 0:
                            percentage = (null_count / data_frame.shape[0]) * 100
                            missing_values_info.append({'Column': col, 'Count': null_count, 'Percentage': percentage})
                    
                    if missing_values_info:
                        st.table(pd.DataFrame(missing_values_info))
                        choose_col = st.selectbox('Choose a Column:', options=[info['Column'] for info in missing_values_info])
                        
                        if choose_col:
                            method = st.selectbox("Choose a Method:", options=['Mean', 'Median', 'Mode', 'Constant', 'Drop the Values (Rows)', 'Drop Entire Column'])
                            
                            if method == 'Mean':
                                mean_value = user_data_frame[choose_col].mean()
                                data = user_data_frame[choose_col].fillna(mean_value)
                                with st.expander(f'See Preview After Filling with Mean ({mean_value:.2f})'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update the Values in Data Frame')
                                if checkbox:
                                    user_data_frame[choose_col] = user_data_frame[choose_col].fillna(mean_value)
                                    st.success(f"Updated column '{choose_col}' with mean value: {mean_value:.2f}")
                            
                            elif method == 'Median':
                                median_value = user_data_frame[choose_col].median()
                                data = user_data_frame[choose_col].fillna(median_value)
                                with st.expander(f'See Preview After Filling with Median ({median_value:.2f})'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update the Values in Data Frame')
                                if checkbox:
                                    user_data_frame[choose_col] = user_data_frame[choose_col].fillna(median_value)
                                    st.success(f"Updated column '{choose_col}' with median value: {median_value:.2f}")
                            
                            elif method == 'Mode':
                                mode_value = user_data_frame[choose_col].mode()[0]
                                data = user_data_frame[choose_col].fillna(mode_value)
                                with st.expander(f'See Preview After Filling with Mode ({mode_value})'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update the Values in Data Frame')
                                if checkbox:
                                    user_data_frame[choose_col] = user_data_frame[choose_col].fillna(mode_value)
                                    st.success(f"Updated column '{choose_col}' with mode value: {mode_value}")
                            
                            elif method == 'Constant':
                                const_value = st.number_input('Enter a Constant Value:')
                                data = user_data_frame[choose_col].fillna(const_value)
                                with st.expander(f'See Preview After Filling with Constant ({const_value})'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update the Values in Data Frame')
                                if checkbox:
                                    user_data_frame[choose_col] = user_data_frame[choose_col].fillna(const_value)
                                    st.success(f"Updated column '{choose_col}' with constant value: {const_value}")
                            
                            elif method == 'Drop the Values (Rows)':
                                data = user_data_frame.dropna(subset=[choose_col])
                                with st.expander('See Preview After Dropping Rows'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update Data Frame - Drop Rows')
                                if checkbox:
                                    user_data_frame = user_data_frame.dropna(subset=[choose_col])
                                    st.success(f"Dropped rows with missing values in column '{choose_col}'")
                            
                            elif method == 'Drop Entire Column':
                                data = user_data_frame.drop(columns=[choose_col])
                                with st.expander('See Preview After Dropping Column'):
                                    st.dataframe(data)
                                
                                checkbox = st.checkbox('Update Data Frame - Drop Column')
                                if checkbox:
                                    user_data_frame = user_data_frame.drop(columns=[choose_col])
                                    numerical_columns.remove(choose_col)
                                    st.success(f"Dropped column '{choose_col}'")
                            
                            with st.expander('Display Updated Data Frame'):
                                st.dataframe(user_data_frame)
                    else:
                        st.success('No Missing Values in Numerical Data')
        
        with tab3:
            st.header("Data Filtering and Selection")
            
            # Update columns lists based on user modifications
            user_categorical_columns = []
            user_numerical_columns = []
            
            for col in user_data_frame.columns:
                if user_data_frame[col].dtype == "object":
                    user_categorical_columns.append(col)
                else:
                    user_numerical_columns.append(col)
            
            st.subheader('Select Columns to Create Custom DataFrame')
            all_column_names = user_data_frame.columns
            user_select_columns = st.multiselect('Select Columns:', options=all_column_names)
            
            if user_select_columns:
                filtered_df = user_data_frame[user_select_columns]
                st.dataframe(filtered_df)
                
                # Add export functionality
                if st.button('Export Selected Columns to CSV'):
                    tmp_download_link = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=tmp_download_link,
                        file_name="filtered_data.csv",
                        mime="text/csv"
                    )
            
            st.subheader('Filter Data by Categorical Columns')
            if user_categorical_columns:
                user_select_categorical_columns = st.multiselect(
                    'Select Categorical Columns to Filter:',
                    options=user_categorical_columns
                )
                
                data = user_data_frame.copy()
                filter_applied = False
                
                for col in user_select_categorical_columns:
                    option_values = sorted(data[col].unique())
                    user_select_data = st.multiselect(
                        f'Select values for column {col}:',
                        options=option_values
                    )
                    
                    if user_select_data:
                        data = data[data[col].isin(user_select_data)]
                        filter_applied = True
                
                if filter_applied:
                    st.subheader('Filtered DataFrame')
                    st.dataframe(data)
                    
                    # Add export functionality
                    if st.button('Export Filtered Data to CSV'):
                        tmp_download_link = data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=tmp_download_link,
                            file_name="filtered_categorical_data.csv",
                            mime="text/csv"
                        )
            
            st.subheader('Filter Data by Numerical Columns')
            if user_numerical_columns:
                user_select_numerical_columns = st.multiselect(
                    'Select Numerical Columns to Filter:',
                    options=user_numerical_columns
                )
                
                data = user_data_frame.copy()
                filter_applied = False
                
                for col in user_select_numerical_columns:
                    min_val = float(data[col].min())
                    max_val = float(data[col].max())
                    
                    # Handle potential identical min/max values
                    if min_val == max_val:
                        st.info(f"Column {col} has identical min and max values: {min_val}")
                        continue
                    
                    range_val = st.slider(
                        f'Select range for {col}:',
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    
                    data = data[(data[col] >= range_val[0]) & (data[col] <= range_val[1])]
                    filter_applied = True
                
                if filter_applied:
                    st.subheader('Filtered DataFrame')
                    st.dataframe(data)
                    
                    # Add export functionality
                    if st.button('Export Numerically Filtered Data to CSV'):
                        tmp_download_link = data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=tmp_download_link,
                            file_name="filtered_numerical_data.csv",
                            mime="text/csv"
                        )
        
        with tab4:
            st.header("Data Visualization")
            
            # Ensure we have at least one numerical column
            if len(user_numerical_columns) == 0:
                st.warning("No numerical columns found for visualization. Please add numerical data or convert existing columns.")
            else:
                visualization_type = st.selectbox(
                    "Select Visualization Type:",
                    options=["Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Bar Chart", "Pie Chart", "Heatmap", "Violin Plot"]
                )
                
                if visualization_type == "Histogram":
                    hist_col = st.selectbox("Select Column for Histogram:", options=user_numerical_columns)
                    bins = st.slider("Number of Bins:", min_value=5, max_value=100, value=20)
                    
                    fig = px.histogram(
                        user_data_frame, 
                        x=hist_col,
                        nbins=bins,
                        title=f"Histogram of {hist_col}",
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add descriptive statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{user_data_frame[hist_col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{user_data_frame[hist_col].median():.2f}")
                    with col3:
                        st.metric("Standard Deviation", f"{user_data_frame[hist_col].std():.2f}")
                
                elif visualization_type == "Box Plot":
                    box_col = st.selectbox("Select Column for Box Plot:", options=user_numerical_columns)
                    
                    # Option to group by a categorical column
                    group_by = st.checkbox("Group by Categorical Column")
                    
                    if group_by and user_categorical_columns:
                        cat_col = st.selectbox("Select Grouping Column:", options=user_categorical_columns)
                        
                        # Check if there are too many categories
                        if user_data_frame[cat_col].nunique() > 10:
                            st.warning(f"There are {user_data_frame[cat_col].nunique()} unique values in {cat_col}. This may make the plot cluttered.")
                            limit_cats = st.checkbox("Limit to top categories")
                            
                            if limit_cats:
                                top_n = st.slider("Number of top categories:", min_value=2, max_value=10, value=5)
                                top_cats = user_data_frame[cat_col].value_counts().nlargest(top_n).index
                                plot_df = user_data_frame[user_data_frame[cat_col].isin(top_cats)]
                                fig = px.box(
                                    plot_df, 
                                    x=cat_col, 
                                    y=box_col,
                                    title=f"Box Plot of {box_col} grouped by {cat_col} (Top {top_n} categories)"
                                )
                            else:
                                fig = px.box(
                                    user_data_frame, 
                                    x=cat_col, 
                                    y=box_col,
                                    title=f"Box Plot of {box_col} grouped by {cat_col}"
                                )
                        else:
                            fig = px.box(
                                user_data_frame, 
                                x=cat_col, 
                                y=box_col,
                                title=f"Box Plot of {box_col} grouped by {cat_col}"
                            )
                    else:
                        fig = px.box(
                            user_data_frame, 
                            y=box_col,
                            title=f"Box Plot of {box_col}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif visualization_type == "Scatter Plot":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_col = st.selectbox("Select X-axis Column:", options=user_numerical_columns)
                    
                    with col2:
                        y_col = st.selectbox("Select Y-axis Column:", 
                                             options=[col for col in user_numerical_columns if col != x_col] 
                                             if len(user_numerical_columns) > 1 else user_numerical_columns)
                    
                    # Optional color grouping
                    color_by = None
                    if user_categorical_columns:
                        use_color = st.checkbox("Color by Category")
                        if use_color:
                            color_by = st.selectbox("Select Category for Color:", options=user_categorical_columns)
                    
                    # Optional size parameter
                    size_by = None
                    if len(user_numerical_columns) > 2:
                        use_size = st.checkbox("Vary Point Size")
                        if use_size:
                            size_options = [col for col in user_numerical_columns if col not in [x_col, y_col]]
                            if size_options:
                                size_by = st.selectbox("Select Column for Point Size:", options=size_options)
                    
                    fig = px.scatter(
                        user_data_frame, 
                        x=x_col, 
                        y=y_col,
                        color=color_by,
                        size=size_by,
                        hover_name=user_data_frame.index if user_data_frame.index.name else None,
                        title=f"Scatter Plot: {y_col} vs {x_col}"
                    )
                    
                    # Add trendline option
                    add_trendline = st.checkbox("Add Trendline")
                    if add_trendline:
                        fig.update_layout(shapes=[
                            dict(
                                type='line',
                                yref='y', xref='x',
                                x0=user_data_frame[x_col].min(), 
                                y0=np.polyval(np.polyfit(user_data_frame[x_col], user_data_frame[y_col], 1), user_data_frame[x_col].min()),
                                x1=user_data_frame[x_col].max(), 
                                y1=np.polyval(np.polyfit(user_data_frame[x_col], user_data_frame[y_col], 1), user_data_frame[x_col].max()),
                                line=dict(color="red", width=2, dash="dash")
                            )
                        ])
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation
                    st.metric(
                        "Correlation Coefficient", 
                        f"{user_data_frame[x_col].corr(user_data_frame[y_col]):.4f}"
                    )
                
                elif visualization_type == "Line Plot":
                    if len(user_data_frame) < 2:
                        st.warning("Need at least 2 data points for a line plot.")
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_options = user_numerical_columns + user_categorical_columns
                            x_col = st.selectbox("Select X-axis Column:", options=x_options)
                        
                        with col2:
                            y_col = st.selectbox("Select Y-axis Column:", options=user_numerical_columns)
                        
                        # Check if we need to sort by x
                        sort_by_x = st.checkbox("Sort by X-axis values", value=True)
                        
                        # Optional grouping
                        group_by = None
                        if user_categorical_columns:
                            use_grouping = st.checkbox("Group Lines by Category")
                            if use_grouping:
                                group_options = [col for col in user_categorical_columns if col != x_col or x_col not in user_categorical_columns]
                                if group_options:
                                    group_by = st.selectbox("Select Grouping Column:", options=group_options)
                        
                        # Prepare data for plot
                        if sort_by_x:
                            plot_df = user_data_frame.sort_values(by=x_col)
                        else:
                            plot_df = user_data_frame
                        
                        fig = px.line(
                            plot_df, 
                            x=x_col, 
                            y=y_col,
                            color=group_by,
                            markers=True,
                            title=f"Line Plot: {y_col} vs {x_col}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif visualization_type == "Bar Chart":
                    st.subheader("Bar Chart Options")
                    
                    chart_type = st.radio("Select Bar Chart Type:", ["Count", "Value"])
                    
                    if chart_type == "Count" and user_categorical_columns:
                        x_col = st.selectbox("Select Column for Categories:", options=user_categorical_columns)
                        
                        # Check if there are too many categories
                        if user_data_frame[x_col].nunique() > 20:
                            st.warning(f"There are {user_data_frame[x_col].nunique()} unique values. This may make the plot cluttered.")
                            limit_cats = st.checkbox("Limit to top categories")
                            
                            if limit_cats:
                                top_n = st.slider("Number of top categories:", min_value=5, max_value=30, value=10)
                                value_counts = user_data_frame[x_col].value_counts().nlargest(top_n)
                                fig = px.bar(
                                    x=value_counts.index, 
                                    y=value_counts.values,
                                    labels={'x': x_col, 'y': 'Count'},
                                    title=f"Count of {x_col} (Top {top_n} categories)"
                                )
                            else:
                                fig = px.bar(
                                    user_data_frame, 
                                    x=x_col,
                                    title=f"Count of {x_col}"
                                )
                        else:
                            fig = px.bar(
                                user_data_frame, 
                                x=x_col,
                                title=f"Count of {x_col}"
                            )
                    
                    elif chart_type == "Value":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            all_columns = user_categorical_columns + user_numerical_columns
                            x_col = st.selectbox("Select X-axis Column:", options=all_columns)
                        
                        with col2:
                            y_col = st.selectbox("Select Y-axis Column (Value):", options=user_numerical_columns)
                        
                        # Optional color grouping
                        color_by = None
                        if user_categorical_columns:
                            color_options = [col for col in user_categorical_columns if col != x_col or x_col not in user_categorical_columns]
                            if color_options:
                                use_color = st.checkbox("Color by Category")
                                if use_color:
                                    color_by = st.selectbox("Select Category for Color:", options=color_options)
                        
                        # Handle potentially large number of unique x values
                        if x_col in user_categorical_columns and user_data_frame[x_col].nunique() > 20:
                            st.warning(f"There are {user_data_frame[x_col].nunique()} unique values in {x_col}. This may make the plot cluttered.")
                            limit_cats = st.checkbox("Limit to top categories by sum")
                            
                            if limit_cats:
                                top_n = st.slider("Number of top categories:", min_value=5, max_value=30, value=10)
                                grouped_df = user_data_frame.groupby(x_col)[y_col].sum().reset_index()
                                top_cats = grouped_df.nlargest(top_n, y_col)[x_col]
                                plot_df = user_data_frame[user_data_frame[x_col].isin(top_cats)]
                                
                                fig = px.bar(
                                    plot_df, 
                                    x=x_col, 
                                    y=y_col,
                                    color=color_by,
                                    title=f"{y_col} by {x_col} (Top {top_n} categories)"
                                )
                            else:
                                fig = px.bar(
                                    user_data_frame, 
                                    x=x_col, 
                                    y=y_col,
                                    color=color_by,
                                    title=f"{y_col} by {x_col}"
                                )
                        else:
                            fig = px.bar(
                                user_data_frame, 
                                x=x_col, 
                                y=y_col,
                                color=color_by,
                                title=f"{y_col} by {x_col}"
                            )
                    
                    else:
                        st.warning("Need categorical columns for this type of bar chart.")
                        fig = None
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif visualization_type == "Pie Chart":
                    if user_categorical_columns:
                        cat_col = st.selectbox("Select Category Column:", options=user_categorical_columns)
                        
                        # Option to use a value column instead of counts
                        use_values = st.checkbox("Use values instead of counts")
                        
                        if use_values and user_numerical_columns:
                            value_col = st.selectbox("Select Value Column:", options=user_numerical_columns)
                            
                            # Group by category and sum values
                            pie_data = user_data_frame.groupby(cat_col)[value_col].sum().reset_index()
                            
                            # Limit segments if too many categories
                            if len(pie_data) > 10:
                                st.warning(f"There are {len(pie_data)} categories. Pie chart will show top 10 by {value_col} and group others.")
                                top_n = st.slider("Number of top categories to show:", min_value=3, max_value=15, value=10)
                                
                                # Get top N categories
                                top_cats = pie_data.nlargest(top_n, value_col)
                                
                                # Sum the rest as "Others"
                                others_sum = pie_data[~pie_data[cat_col].isin(top_cats[cat_col])][value_col].sum()
                                
                                if others_sum > 0:
                                    others_df = pd.DataFrame({cat_col: ['Others'], value_col: [others_sum]})
                                    pie_data = pd.concat([top_cats, others_df])
                            
                            fig = px.pie(
                                pie_data, 
                                names=cat_col, 
                                values=value_col,
                                title=f"Distribution of {value_col} by {cat_col}"
                            )
                        
                        else:
                            # Use value counts directly
                            value_counts = user_data_frame[cat_col].value_counts()
                            
                            # Limit segments if too many categories
                            if len(value_counts) > 10:
                                st.warning(f"There are {len(value_counts)} categories. Pie chart will show top 10 and group others.")
                                top_n = st.slider("Number of top categories to show:", min_value=3, max_value=15, value=10)
                                
                                # Get top N categories
                                top_cats = value_counts.nlargest(top_n)
                                
                                # Sum the rest as "Others"
                                others_count = value_counts[~value_counts.index.isin(top_cats.index)].sum()
                                
                                if others_count > 0:
                                    pie_data = pd.concat([top_cats, pd.Series([others_count], index=['Others'])])
                                else:
                                    pie_data = top_cats
                            else:
                                pie_data = value_counts
                            
                            fig = px.pie(
                                values=pie_data.values, 
                                names=pie_data.index, 
                                title=f"Distribution of {cat_col}"
                            )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning("Need categorical columns for a pie chart.")
                
                elif visualization_type == "Heatmap":
                    st.subheader("Correlation Heatmap")
                    
                    if len(user_numerical_columns) < 2:
                        st.warning("Need at least 2 numerical columns for a correlation heatmap.")
                    else:
                        # User can select a subset of columns
                        selected_cols = st.multiselect(
                            "Select columns for correlation (default: all numerical):", 
                            options=user_numerical_columns,
                            default=user_numerical_columns if len(user_numerical_columns) <= 10 else user_numerical_columns[:10]
                        )
                        
                        if not selected_cols:
                            selected_cols = user_numerical_columns
                        
                        # Create correlation matrix
                        corr_matrix = user_data_frame[selected_cols].corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            aspect="auto",
                            title="Correlation Heatmap"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Identify strongest correlations
                        if len(selected_cols) > 2:
                            st.subheader("Strongest Correlations")
                            
                            # Get upper triangle only (to avoid duplicates)
                            upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                            
                            # Stack and sort
                            strongest_corr = upper_corr.stack().reset_index()
                            strongest_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
                            strongest_corr = strongest_corr.sort_values('Correlation', key=abs, ascending=False)
                            
                            # Display top correlations
                            st.table(strongest_corr.head(10))
                
                elif visualization_type == "Violin Plot":
                    y_col = st.selectbox("Select Numerical Column:", options=user_numerical_columns)
                    
                    # Optional grouping
                    use_group = False
                    if user_categorical_columns:
                        use_group = st.checkbox("Group by Categorical Variable")
                    
                    if use_group and user_categorical_columns:
                        x_col = st.selectbox("Select Grouping Column:", options=user_categorical_columns)
                        
                        # Check if there are too many categories
                        if user_data_frame[x_col].nunique() > 10:
                            st.warning(f"There are {user_data_frame[x_col].nunique()} unique values in {x_col}. This may make the plot cluttered.")
                            limit_cats = st.checkbox("Limit to top categories by count")
                            
                            if limit_cats:
                                top_n = st.slider("Number of top categories:", min_value=2, max_value=10, value=5)
                                top_cats = user_data_frame[x_col].value_counts().nlargest(top_n).index
                                plot_df = user_data_frame[user_data_frame[x_col].isin(top_cats)]
                                
                                fig = px.violin(
                                    plot_df, 
                                    x=x_col, 
                                    y=y_col,
                                    box=True,
                                    points="all",
                                    title=f"Distribution of {y_col} by {x_col} (Top {top_n} categories)"
                                )
                            else:
                                fig = px.violin(
                                    user_data_frame, 
                                    x=x_col, 
                                    y=y_col,
                                    box=True,
                                    points="all",
                                    title=f"Distribution of {y_col} by {x_col}"
                                )
                        else:
                            fig = px.violin(
                                user_data_frame, 
                                x=x_col, 
                                y=y_col,
                                box=True,
                                points="all",
                                title=f"Distribution of {y_col} by {x_col}"
                            )
                    else:
                        fig = px.violin(
                            user_data_frame, 
                            y=y_col,
                            box=True,
                            points="all",
                            title=f"Distribution of {y_col}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.header("Correlation Analysis")
            
            if len(user_numerical_columns) < 2:
                st.warning("Need at least 2 numerical columns for correlation analysis.")
            else:
                # Correlation type
                corr_method = st.radio(
                    "Select Correlation Method:",
                    options=["Pearson", "Spearman"],
                    help="Pearson measures linear correlation, Spearman measures monotonic relationship"
                )
                
                # User can select a subset of columns
                selected_cols = st.multiselect(
                    "Select columns for correlation analysis:", 
                    options=user_numerical_columns,
                    default=user_numerical_columns if len(user_numerical_columns) <= 10 else user_numerical_columns[:10]
                )
                
                if not selected_cols:
                    selected_cols = user_numerical_columns
                
                # Create correlation matrix
                corr_matrix = user_data_frame[selected_cols].corr(method=corr_method.lower())
                
                # Display correlation matrix
                st.subheader("Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.4f}"))
                
                # Create heatmap visualization
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title=f"{corr_method} Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Identify strongest correlations
                if len(selected_cols) > 2:
                    st.subheader("Strongest Correlations")
                    
                    # Get upper triangle only (to avoid duplicates)
                    upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    
                    # Stack and sort
                    strongest_corr = upper_corr.stack().reset_index()
                    strongest_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
                    strongest_corr = strongest_corr.sort_values('Correlation', key=abs, ascending=False)
                    
                    # Display top correlations
                    st.table(strongest_corr.head(10))
                    
                    # Visualize top correlation with scatter plot
                    if not strongest_corr.empty:
                        st.subheader("Scatter Plot of Strongest Correlation")
                        top_pair = strongest_corr.iloc[0]
                        var1, var2 = top_pair['Variable 1'], top_pair['Variable 2']
                        
                        fig = px.scatter(
                            user_data_frame, 
                            x=var1, 
                            y=var2,
                            trendline="ols",
                            title=f"Strongest Correlation: {var1} vs {var2} (r = {top_pair['Correlation']:.4f})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed exploration of specific correlation
                st.subheader("Explore Specific Correlation")
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Select First Variable:", options=selected_cols)
                
                with col2:
                    remaining_cols = [col for col in selected_cols if col != var1]
                    var2 = st.selectbox("Select Second Variable:", options=remaining_cols if remaining_cols else selected_cols)
                
                if var1 != var2:
                    correlation = user_data_frame[var1].corr(user_data_frame[var2], method=corr_method.lower())
                    
                    st.metric(
                        f"{corr_method} Correlation Coefficient", 
                        f"{correlation:.4f}"
                    )
                    
                    fig = px.scatter(
                        user_data_frame, 
                        x=var1, 
                        y=var2,
                        trendline="ols",
                        title=f"{var1} vs {var2} (r = {correlation:.4f})"
                    )
                    
                    # Add categorical color option
                    if user_categorical_columns:
                        color_by = st.selectbox("Color points by (optional):", options=["None"] + user_categorical_columns)
                        if color_by != "None":
                            fig = px.scatter(
                                user_data_frame, 
                                x=var1, 
                                y=var2,
                                color=color_by,
                                trendline="ols",
                                title=f"{var1} vs {var2} (r = {correlation:.4f})"
                            )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add regression summary
                    import statsmodels.api as sm
                    
                    X = sm.add_constant(user_data_frame[var1])
                    y = user_data_frame[var2]
                    
                    try:
                        model = sm.OLS(y, X).fit()
                        with st.expander("Show Regression Analysis"):
                            st.text(model.summary().as_text())
                            
                            # Simplified interpretation
                            slope = model.params[var1]
                            intercept = model.params['const']
                            r_squared = model.rsquared
                            p_value = model.pvalues[var1]
                            
                            st.write(f"**Linear Equation**: {var2} = {intercept:.4f} + {slope:.4f} × {var1}")
                            st.write(f"**R-squared**: {r_squared:.4f} (explains {r_squared*100:.1f}% of variance)")
                            
                            # Significance interpretation
                            if p_value < 0.001:
                                significance = "highly significant"
                            elif p_value < 0.01:
                                significance = "very significant"
                            elif p_value < 0.05:
                                significance = "significant"
                            elif p_value < 0.1:
                                significance = "marginally significant"
                            else:
                                significance = "not significant"
                            
                            st.write(f"**Significance**: The relationship is {significance} (p = {p_value:.4f})")
                    except:
                        st.warning("Could not perform regression analysis. There might be missing values or other issues.")
        
        with tab6:
            st.header("Advanced Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                options=["Descriptive Statistics", "PCA (Principal Component Analysis)", "Outlier Detection"]
            )
            
            if analysis_type == "Descriptive Statistics":
                st.subheader("Detailed Statistical Analysis")
                
                # Select columns for analysis
                analysis_cols = st.multiselect(
                    "Select columns for detailed analysis:",
                    options=user_data_frame.columns.tolist(),
                    default=user_numerical_columns[:5] if len(user_numerical_columns) > 5 else user_numerical_columns
                )
                
                if analysis_cols:
                    # Basic statistics
                    stats_df = user_data_frame[analysis_cols].describe(include='all').T
                    
                    # Add additional statistics for numerical columns
                    for col in analysis_cols:
                        if col in user_numerical_columns:
                            stats_df.loc[col, 'skewness'] = user_data_frame[col].skew()
                            stats_df.loc[col, 'kurtosis'] = user_data_frame[col].kurtosis()
                            stats_df.loc[col, 'IQR'] = user_data_frame[col].quantile(0.75) - user_data_frame[col].quantile(0.25)
                            stats_df.loc[col, 'CV (%)'] = (user_data_frame[col].std() / user_data_frame[col].mean() * 100) if user_data_frame[col].mean() != 0 else np.nan
                    
                    st.dataframe(stats_df)
                    
                    # Individual column analysis
                    for col in analysis_cols:
                        with st.expander(f"Detailed Analysis of {col}"):
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                if col in user_numerical_columns:
                                    st.write("**Five Number Summary**")
                                    five_num = {
                                        "Minimum": user_data_frame[col].min(),
                                        "1st Quartile": user_data_frame[col].quantile(0.25),
                                        "Median": user_data_frame[col].median(),
                                        "3rd Quartile": user_data_frame[col].quantile(0.75),
                                        "Maximum": user_data_frame[col].max()
                                    }
                                    st.dataframe(pd.Series(five_num, name=col))
                                    
                                    st.write("**Distribution Metrics**")
                                    dist_metrics = {
                                        "Mean": user_data_frame[col].mean(),
                                        "Standard Deviation": user_data_frame[col].std(),
                                        "Skewness": user_data_frame[col].skew(),
                                        "Kurtosis": user_data_frame[col].kurtosis()
                                    }
                                    st.dataframe(pd.Series(dist_metrics, name=col))
                                    
                                    # Interpretation of skewness and kurtosis
                                    skew = user_data_frame[col].skew()
                                    kurt = user_data_frame[col].kurtosis()
                                    
                                    skew_interp = ""
                                    if abs(skew) < 0.5:
                                        skew_interp = "approximately symmetric"
                                    elif abs(skew) < 1:
                                        skew_interp = "moderately skewed"
                                    else:
                                        skew_interp = "highly skewed"
                                        
                                    if skew > 0:
                                        skew_interp += " to the right"
                                    elif skew < 0:
                                        skew_interp += " to the left"
                                    
                                    kurt_interp = ""
                                    if kurt < -0.5:
                                        kurt_interp = "platykurtic (flatter than normal)"
                                    elif kurt > 0.5:
                                        kurt_interp = "leptokurtic (more peaked than normal)"
                                    else:
                                        kurt_interp = "mesokurtic (similar to normal)"
                                    
                                    st.write(f"Distribution is {skew_interp} and {kurt_interp}.")
                                    
                                else:  # Categorical column
                                    value_counts = user_data_frame[col].value_counts()
                                    st.write("**Frequency Distribution**")
                                    st.dataframe(value_counts)
                                    
                                    st.write("**Proportions**")
                                    st.dataframe((value_counts / len(user_data_frame) * 100).round(2).rename("Percentage (%)"))
                                    
                                    most_common = value_counts.index[0] if not value_counts.empty else None
                                    st.write(f"Most common value: **{most_common}** ({value_counts.iloc[0]} occurrences)")
                            
                            with col2:
                                if col in user_numerical_columns:
                                    # Create histogram and box plot
                                    fig = go.Figure()
                                    
                                    # Add histogram
                                    fig.add_trace(go.Histogram(
                                        x=user_data_frame[col],
                                        name="Histogram",
                                        opacity=0.7
                                    ))
                                    
                                    # Add kernel density estimate
                                    try:
                                        from scipy import stats
                                        kde_x = np.linspace(user_data_frame[col].min(), user_data_frame[col].max(), 1000)
                                        kde = stats.gaussian_kde(user_data_frame[col].dropna())
                                        kde_y = kde(kde_x)
                                        
                                        # Scale KDE to match histogram
                                        hist, bin_edges = np.histogram(user_data_frame[col].dropna(), bins='auto')
                                        scaling_factor = np.max(hist) / np.max(kde_y) if np.max(kde_y) > 0 else 1
                                        
                                        fig.add_trace(go.Scatter(
                                            x=kde_x,
                                            y=kde_y * scaling_factor,
                                            mode='lines',
                                            name='KDE',
                                            line=dict(color='red', width=2)
                                        ))
                                    except:
                                        st.warning("Could not compute kernel density estimate.")
                                    
                                    fig.update_layout(
                                        title=f"Distribution of {col}",
                                        xaxis_title=col,
                                        yaxis_title="Frequency",
                                        bargap=0.05
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Box plot
                                    fig = px.box(
                                        user_data_frame, 
                                        y=col,
                                        points="all",
                                        notched=True,
                                        title=f"Box Plot of {col}"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                else:  # Categorical column
                                    # Bar chart of frequencies
                                    value_counts = user_data_frame[col].value_counts()
                                    
                                    if len(value_counts) > 15:
                                        # Show only top values
                                        top_n = 15
                                        fig = px.bar(
                                            x=value_counts.nlargest(top_n).index,
                                            y=value_counts.nlargest(top_n).values,
                                            title=f"Top {top_n} values for {col}",
                                            labels={'x': col, 'y': 'Frequency'}
                                        )
                                    else:
                                        fig = px.bar(
                                            x=value_counts.index,
                                            y=value_counts.values,
                                            title=f"Frequency of {col}",
                                            labels={'x': col, 'y': 'Frequency'}
                                        )
                                    
                                    fig.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical tests
                    if len(user_numerical_columns) >= 1:
                        with st.expander("Statistical Tests"):
                            st.subheader("Normality Tests")
                            st.write("Testing if the data follows a normal distribution.")
                            
                            test_cols = st.multiselect(
                                "Select columns for normality test:",
                                options=user_numerical_columns,
                                default=user_numerical_columns[0] if user_numerical_columns else None
                            )
                            
                            if test_cols:
                                from scipy import stats
                                
                                results = []
                                for col in test_cols:
                                    # Skip columns with too few unique values
                                    if user_data_frame[col].nunique() < 10:
                                        results.append({
                                            "Column": col,
                                            "Test": "Shapiro-Wilk",
                                            "Statistic": None,
                                            "p-value": None,
                                            "Normal": "Too few unique values"
                                        })
                                        continue
                                    
                                    # Sampling for large datasets
                                    sample_data = user_data_frame[col].dropna()
                                    if len(sample_data) > 5000:
                                        st.info(f"The dataset is large. Using a random sample of 5000 rows for the Shapiro-Wilk test on {col}.")
                                        sample_data = sample_data.sample(5000)
                                    
                                    try:
                                        stat, p = stats.shapiro(sample_data)
                                        results.append({
                                            "Column": col,
                                            "Test": "Shapiro-Wilk",
                                            "Statistic": stat,
                                            "p-value": p,
                                            "Normal": "Yes" if p > 0.05 else "No"
                                        })
                                    except Exception as e:
                                        results.append({
                                            "Column": col,
                                            "Test": "Shapiro-Wilk",
                                            "Statistic": None,
                                            "p-value": None,
                                            "Normal": f"Test failed: {str(e)}"
                                        })
                                
                                st.dataframe(pd.DataFrame(results))
                                
                                st.write("""
                                **Interpretation**:
                                - p-value > 0.05: We cannot reject the null hypothesis that the data is normally distributed.
                                - p-value ≤ 0.05: We reject the null hypothesis - the data is not normally distributed.
                                """)
            
            elif analysis_type == "PCA (Principal Component Analysis)":
                if len(user_numerical_columns) < 2:
                    st.warning("PCA requires at least 2 numerical columns. Please add more numerical data.")
                else:
                    st.subheader("Principal Component Analysis (PCA)")
                    st.write("""
                    PCA is a dimensionality reduction technique that transforms the data into a new coordinate system
                    where the greatest variances lie on the first few principal components.
                    """)
                    
                    # Select columns for PCA
                    pca_cols = st.multiselect(
                        "Select numerical columns for PCA:",
                        options=user_numerical_columns,
                        default=user_numerical_columns[:min(5, len(user_numerical_columns))]
                    )
                    
                    if len(pca_cols) >= 2:
                        # Number of components
                        n_components = st.slider(
                            "Number of principal components:", 
                            min_value=2, 
                            max_value=min(len(pca_cols), 10),
                            value=min(3, len(pca_cols))
                        )
                        
                        # Get data and handle missing values
                        pca_data = user_data_frame[pca_cols].copy()
                        
                        # Check for missing values
                        if pca_data.isnull().sum().sum() > 0:
                            st.warning("The selected data contains missing values. These will be filled with the mean for PCA.")
                            pca_data = pca_data.fillna(pca_data.mean())
                        
                        # Standardize the data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)
                        
                        # Perform PCA
                        pca = PCA(n_components=n_components)
                        principal_components = pca.fit_transform(scaled_data)
                        
                        # Create DataFrame with principal components
                        pc_df = pd.DataFrame(
                            data=principal_components,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        # Display variance explained
                        explained_variance = pca.explained_variance_ratio_ * 100
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        variance_df = pd.DataFrame({
                            'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                            'Variance Explained (%)': explained_variance,
                            'Cumulative Variance (%)': cumulative_variance
                        })
                        
                        st.subheader("Variance Explained by Principal Components")
                        st.dataframe(variance_df)
                        
                        # Plot variance explained
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=variance_df['Principal Component'],
                            y=variance_df['Variance Explained (%)'],
                            name='Individual'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=variance_df['Principal Component'],
                            y=variance_df['Cumulative Variance (%)'],
                            mode='lines+markers',
                            name='Cumulative',
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title="Variance Explained by Principal Components",
                            xaxis_title="Principal Component",
                            yaxis_title="Variance Explained (%)",
                            yaxis2=dict(
                                title="Cumulative Variance (%)",
                                overlaying="y",
                                side="right"
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display loadings
                        loadings = pca.components_
                        loadings_df = pd.DataFrame(
                            data=loadings.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=pca_cols
                        )
                        
                        st.subheader("Component Loadings (Feature Weights)")
                        st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm').format("{:.4f}"))
                        
                        # Visualize the loadings
                        if len(pca_cols) <= 20:  # Only show if not too many features
                            fig = px.imshow(
                                loadings_df,
                                labels=dict(x="Principal Component", y="Feature", color="Loading"),
                                x=loadings_df.columns,
                                y=loadings_df.index,
                                color_continuous_scale='RdBu_r',
                                aspect="auto",
                                title="PCA Loadings Heatmap"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 2D Scatter plot of first two PCs
                        st.subheader("Data Projected onto Principal Components")
                        
                        if n_components >= 2:
                            # Color by category if available
                            color_by = None
                            if user_categorical_columns:
                                color_option = st.selectbox(
                                    "Color points by (optional):", 
                                    options=["None"] + user_categorical_columns
                                )
                                if color_option != "None":
                                    color_by = user_data_frame[color_option]
                            
                            # Plot
                            fig = px.scatter(
                                pc_df, 
                                x='PC1', 
                                y='PC2',
                                color=color_by,
                                labels={'color': color_option if color_by is not None else None},
                                title="PCA: First Two Principal Components",
                                opacity=0.7
                            )
                            
                            # Add loadings as vectors (biplot)
                            show_biplot = st.checkbox("Show feature vectors (biplot)")
                            
                            if show_biplot:
                                # Scale the loadings
                                scalex = 1.0 / (pc_df['PC1'].max() - pc_df['PC1'].min())
                                scaley = 1.0 / (pc_df['PC2'].max() - pc_df['PC2'].min())
                                
                                for i, feature in enumerate(pca_cols):
                                    fig.add_shape(
                                        type='line',
                                        x0=0, y0=0,
                                        x1=loadings[0, i] / scalex,
                                        y1=loadings[1, i] / scaley,
                                        line=dict(color='red', width=1),
                                        name=feature
                                    )
                                    
                                    fig.add_annotation(
                                        x=loadings[0, i] / scalex,
                                        y=loadings[1, i] / scaley,
                                        ax=0, ay=0,
                                        xanchor="center",
                                        yanchor="bottom",
                                        text=feature,
                                        showarrow=False
                                    )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 3D plot if we have at least 3 components
                        if n_components >= 3:
                            st.subheader("3D Visualization of First Three Principal Components")
                            
                            # Color by category if available
                            color_by = None
                            if user_categorical_columns:
                                color_option = st.selectbox(
                                    "Color points by (optional) for 3D plot:", 
                                    options=["None"] + user_categorical_columns,
                                    key="color_3d"
                                )
                                if color_option != "None":
                                    color_by = user_data_frame[color_option]
                            
                            fig = px.scatter_3d(
                                pc_df, 
                                x='PC1', 
                                y='PC2', 
                                z='PC3',
                                color=color_by,
                                labels={'color': color_option if color_by is not None else None},
                                opacity=0.7,
                                title="PCA: First Three Principal Components"
                            )
                            
                            fig.update_layout(
                                scene=dict(
                                    xaxis_title='PC1',
                                    yaxis_title='PC2',
                                    zaxis_title='PC3'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least 2 columns for PCA.")
            
            elif analysis_type == "Outlier Detection":
                st.subheader("Outlier Detection")
                
                # Select columns for outlier detection
                outlier_cols = st.multiselect(
                    "Select numerical columns for outlier detection:",
                    options=user_numerical_columns,
                    default=user_numerical_columns[0] if user_numerical_columns else None
                )
                
                if outlier_cols:
                    # Method for outlier detection
                    method = st.radio(
                        "Select outlier detection method:",
                        options=["Z-Score", "IQR (Interquartile Range)"],
                        index=1,
                        help="Z-Score identifies values far from the mean. IQR identifies values far from the median."
                    )
                    
                    # Parameters based on method
                    if method == "Z-Score":
                        threshold = st.slider(
                            "Z-Score threshold:", 
                            min_value=1.0, 
                            max_value=5.0, 
                            value=3.0, 
                            step=0.1,
                            help="Values with absolute Z-score above this threshold will be considered outliers."
                        )
                    else:  # IQR
                        iqr_factor = st.slider(
                            "IQR factor:", 
                            min_value=1.0, 
                            max_value=3.0, 
                            value=1.5, 
                            step=0.1,
                            help="Values outside Q1 - factor*IQR and Q3 + factor*IQR will be considered outliers."
                        )
                    
                    # Process each selected column
                    all_outliers = pd.DataFrame()
                    
                    for col in outlier_cols:
                        data = user_data_frame[col].dropna()
                        
                        if method == "Z-Score":
                            z_scores = np.abs((data - data.mean()) / data.std())
                            outliers = user_data_frame[z_scores > threshold].copy()
                            
                            if not outliers.empty:
                                outliers['Z-Score'] = z_scores[z_scores > threshold]
                                outliers['Outlier_Method'] = 'Z-Score'
                                outliers['Column'] = col
                                all_outliers = pd.concat([all_outliers, outliers])
                        
                        else:  # IQR method
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            
                            lower_bound = Q1 - iqr_factor * IQR
                            upper_bound = Q3 + iqr_factor * IQR
                            
                            outliers = user_data_frame[(data < lower_bound) | (data > upper_bound)].copy()
                            
                            if not outliers.empty:
                                outliers['Boundary'] = np.where(
                                    data[outliers.index] < lower_bound, 
                                    f'< {lower_bound:.2f}', 
                                    f'> {upper_bound:.2f}'
                                )
                                outliers['Outlier_Method'] = 'IQR'
                                outliers['Column'] = col
                                all_outliers = pd.concat([all_outliers, outliers])
                    
                    # Display results
                    if all_outliers.empty:
                        st.success("No outliers detected with the current settings.")
                    else:
                        st.write(f"Found {len(all_outliers)} potential outliers across all selected columns.")
                        
                        # Group by column
                        for col in outlier_cols:
                            col_outliers = all_outliers[all_outliers['Column'] == col]
                            
                            if not col_outliers.empty:
                                with st.expander(f"Outliers in {col} ({len(col_outliers)} found)"):
                                    st.dataframe(col_outliers)
                                    
                                    # Visualize
                                    fig = px.box(
                                        user_data_frame, 
                                        y=col,
                                        points="all",
                                        title=f"Box Plot with Outliers: {col}"
                                    )
                                    
                                    # Highlight outliers
                                    outlier_points = go.Scatter(
                                        y=col_outliers[col],
                                        mode='markers',
                                        marker=dict(
                                            color='red',
                                            size=10,
                                            symbol='circle-open',
                                            line=dict(width=2)
                                        ),
                                        name='Outliers'
                                    )
                                    
                                    fig.add_trace(outlier_points)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to remove outliers
                        st.subheader("Handle Outliers")
                        action = st.radio(
                            "Select action for outliers:",
                            options=["Keep outliers", "Remove outliers", "Replace with bounds", "Replace with mean/median"]
                        )
                        
                        if action != "Keep outliers":
                            # Preview the result
                            modified_df = user_data_frame.copy()
                            
                            if action == "Remove outliers":
                                modified_df = modified_df.drop(index=all_outliers.index.unique())
                                
                                st.write(f"After removing outliers: {len(modified_df)} rows (removed {len(user_data_frame) - len(modified_df)} rows)")
                            
                            elif action == "Replace with bounds":
                                for col in outlier_cols:
                                    data = modified_df[col].dropna()
                                    
                                    if method == "Z-Score":
                                        z_scores = np.abs((data - data.mean()) / data.std())
                                        mean = data.mean()
                                        std = data.std()
                                        
                                        # Replace values beyond threshold
                                        high_mask = (z_scores > threshold) & (data > mean)
                                        low_mask = (z_scores > threshold) & (data < mean)
                                        
                                        modified_df.loc[high_mask.index[high_mask], col] = mean + threshold * std
                                        modified_df.loc[low_mask.index[low_mask], col] = mean - threshold * std
                                    
                                    else:  # IQR method
                                        Q1 = data.quantile(0.25)
                                        Q3 = data.quantile(0.75)
                                        IQR = Q3 - Q1
                                        
                                        lower_bound = Q1 - iqr_factor * IQR
                                        upper_bound = Q3 + iqr_factor * IQR
                                        
                                        # Replace values beyond bounds
                                        modified_df.loc[data.index[data < lower_bound], col] = lower_bound
                                        modified_df.loc[data.index[data > upper_bound], col] = upper_bound
                                
                                st.write("Replaced outliers with boundary values")
                            
                            elif action == "Replace with mean/median":
                                replacement = st.radio(
                                    "Replace with:",
                                    options=["Mean", "Median"]
                                )
                                
                                for col in outlier_cols:
                                    data = modified_df[col].dropna()
                                    
                                    # Find outlier indices
                                    if method == "Z-Score":
                                        z_scores = np.abs((data - data.mean()) / data.std())
                                        outlier_indices = data.index[z_scores > threshold]
                                    else:  # IQR method
                                        Q1 = data.quantile(0.25)
                                        Q3 = data.quantile(0.75)
                                        IQR = Q3 - Q1
                                        
                                        lower_bound = Q1 - iqr_factor * IQR
                                        upper_bound = Q3 + iqr_factor * IQR
                                        
                                        outlier_indices = data.index[(data < lower_bound) | (data > upper_bound)]
                                    
                                    # Replace values
                                    if replacement == "Mean":
                                        modified_df.loc[outlier_indices, col] = data.mean()
                                    else:  # Median
                                        modified_df.loc[outlier_indices, col] = data.median()
                                
                                st.write(f"Replaced outliers with {replacement.lower()} values")
                            
                            # Preview the result
                            with st.expander("Preview data after handling outliers"):
                                st.dataframe(modified_df)
                            
                            # Apply changes
                            if st.button("Apply outlier handling to the dataset"):
                                user_data_frame = modified_df.copy()
                                st.success("Successfully applied outlier handling to the dataset.")
                                st.dataframe(user_data_frame)
        # Database Tab
        with tab7:
            st.header("Database Operations")
            
            db_operation = st.radio(
                "Select Operation:",
                options=["Save Current Dataset", "Load Saved Dataset", "View All Saved Datasets", "Delete Dataset"]
            )
            
            if db_operation == "Save Current Dataset":
                st.subheader("Save Current Dataset to Database")
                
                if user_data_frame is not None:
                    dataset_name = st.text_input("Dataset Name:", value=uploaded_file.name if uploaded_file else "My Dataset")
                    dataset_description = st.text_area("Description (optional):", value="")
                    
                    if st.button("Save Dataset"):
                        try:
                            dataset_id = db.save_dataframe(dataset_name, user_data_frame, dataset_description)
                            st.success(f"Dataset successfully saved with ID: {dataset_id}")
                        except Exception as db_error:
                            st.error(f"Error saving dataset: {str(db_error)}")
                else:
                    st.warning("Please upload and process a dataset first.")
            
            elif db_operation == "Load Saved Dataset":
                st.subheader("Load Dataset from Database")
                
                try:
                    datasets = db.get_all_datasets()
                    if datasets:
                        dataset_options = {f"{d['name']} (ID: {d['id']})": d['id'] for d in datasets}
                        selected_dataset = st.selectbox("Select Dataset to Load:", options=list(dataset_options.keys()))
                        
                        if st.button("Load Selected Dataset"):
                            dataset_id = dataset_options[selected_dataset]
                            data_frame = db.get_dataframe(dataset_id)
                            user_data_frame = data_frame.copy()
                            
                            # Recategorize columns
                            categorical_columns = []
                            numerical_columns = []
                            for col in data_frame.columns:
                                if data_frame[col].dtype == "object":
                                    categorical_columns.append(col)
                                else:
                                    numerical_columns.append(col)
                            
                            st.success(f"Dataset '{selected_dataset}' loaded successfully")
                            st.dataframe(data_frame)
                            st.session_state['data_loaded_from_db'] = True
                    else:
                        st.info("No datasets found in the database.")
                except Exception as db_error:
                    st.error(f"Error loading datasets: {str(db_error)}")
            
            elif db_operation == "View All Saved Datasets":
                st.subheader("All Saved Datasets")
                
                try:
                    datasets = db.get_all_datasets()
                    if datasets:
                        datasets_df = pd.DataFrame(datasets)
                        st.dataframe(datasets_df)
                        
                        # Show dataset statistics
                        st.subheader("Dataset Statistics")
                        st.metric("Total Saved Datasets", len(datasets))
                        
                        # Create a bar chart of row counts
                        fig = px.bar(
                            datasets_df, 
                            x='name', 
                            y='row_count',
                            title='Dataset Sizes (Row Count)',
                            labels={'name': 'Dataset Name', 'row_count': 'Number of Rows'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No datasets found in the database.")
                except Exception as db_error:
                    st.error(f"Error retrieving datasets: {str(db_error)}")
            
            elif db_operation == "Delete Dataset":
                st.subheader("Delete Dataset from Database")
                st.warning("⚠️ This operation cannot be undone!")
                
                try:
                    datasets = db.get_all_datasets()
                    if datasets:
                        dataset_options = {f"{d['name']} (ID: {d['id']})": d['id'] for d in datasets}
                        selected_dataset = st.selectbox("Select Dataset to Delete:", options=list(dataset_options.keys()))
                        
                        if st.button("Delete Selected Dataset"):
                            dataset_id = dataset_options[selected_dataset]
                            if db.delete_dataset(dataset_id):
                                st.success(f"Dataset '{selected_dataset}' deleted successfully")
                            else:
                                st.error("Error deleting dataset")
                    else:
                        st.info("No datasets found in the database.")
                except Exception as db_error:
                    st.error(f"Error: {str(db_error)}")
            
            st.markdown("---")
            st.info("The database stores your datasets securely, allowing you to access them in future sessions.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check if your CSV file is properly formatted and try again.")

# Display author information footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    ### Data Analysis Dashboard
    A comprehensive tool for data analysis without writing code.
    
    **Features**:
    - Upload and analyze CSV files
    - Clean and transform data
    - Create visualizations
    - Perform statistical analysis
    - Save and load datasets from database
    - Export results
    
    Made with ❤️ using Streamlit
    """
)
