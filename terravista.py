# Importing required packages
import pandas as pd
import streamlit as st
import plotly.express as px
import traceback
import numpy as np

# Setting layout to be wide
st.set_page_config(layout="wide")

# Function to load the .csv data of differeing encodings
@st.cache_data # caching so loading only occurs once. Note that cache persists for duraction of session unless manually cleared
def loaddata(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0) # Set the seek function back to start after each attempt, otherwise incrementally increases
        
        try:
            file = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                file = pd.read_csv(uploaded_file, encoding='latin1')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    file = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                except UnicodeDecodeError:
                    st.error("Unable to read the file with UTF-8, Latin-1, or ISO-8859-1 encoding. Please check the file encoding.")
                    return pd.DataFrame()
        return file
    else:
        print("Please upload a file.")
        return pd.DataFrame()
        
# Creating a list of the column headers for use as variables to filter on
def createvariables(inputdata):
    if not inputdata.empty:
        variables = inputdata.columns
        variables = list(variables)
        return variables
    else:
        st.warning("No data available to create variables.")
        return []

# Fucntion to allow user to define the filters to be applied to dataset
def selectvariables(inputdata):
    filters = createvariables(inputdata)
    if filters:
        userselection = st.multiselect("What do you want to filter on?", options=filters)
        return userselection
    else:
        return []

# For each variable which is selected as a filter, allow user to select wheter value, contains or range for each
def filterdata(filters, data):
    for i in filters:
        value_or_range_or_contains = st.radio("Value, Range, or Contains?", ("Value", "Range", "Contains"), horizontal=True, key=i)
        if value_or_range_or_contains == "Value":
            choices = data[i].unique()
            choices_with_select_all = ["Select All"] + list(choices)
            user_selection = st.multiselect(f"{i} Selection", options=choices_with_select_all)
            # Create a user to select all values - for cases where multislecting would be a total bore due to size of data
            if "Select All" in user_selection:
                data = data
            else:
                data = data[data[i].isin(user_selection)]
        elif value_or_range_or_contains == "Range":
            min_value = float(min(data[i]))
            max_value = float(max(data[i]))
            lowerbound = st.text_input(f"Set Lower Bound for {i}", value=str(min_value))
            upperbound = st.text_input(f"Set Upper Bound for {i}", value=str(max_value))
            try:
                lowerbound = float(lowerbound)
                upperbound = float(upperbound)
                if lowerbound < min_value or lowerbound > max_value:
                    st.error(f"Lower bound must be between {min_value} and {max_value}.")
                elif upperbound < min_value or upperbound > max_value:
                    st.error(f"Upper bound must be between {min_value} and {max_value}.")
                elif lowerbound >= upperbound:
                    st.error("Lower bound must be less than upper cutoff.")
                else:
                    st.success(f"Range successfully set from {lowerbound} to {upperbound}.")
                    data = data[(data[i] >= lowerbound) & (data[i] <= upperbound)]
            except ValueError:
                st.error(f"Please enter valid numeric values for both range bounds. i.e., between {min_value} and {max_value}.")
        elif value_or_range_or_contains == "Contains":
            contains_value = st.text_input(f"Enter value or letter for {i} to contain")
            if contains_value:
                data = data[data[i].astype(str).str.contains(contains_value, case=False, na=False)]
    return data

# Plot single or multiple variables with respect to depth
def createdownholeplots(data, holeid_col, from_col, to_col):
    selected_analytes = st.multiselect("Select variable to plot", options=data.columns)
    
    exclusions = [holeid_col, from_col, to_col, selected_analytes]
    hover_data_options = st.multiselect("Select hover data", options=[col for col in data.columns if col not in exclusions])
    
    data[from_col] = pd.to_numeric(data[from_col], errors='coerce')
    data[to_col] = pd.to_numeric(data[to_col], errors='coerce')
    
    # Using the midpoint of an interval to make plotting of variable values easier
    data['Interval Midpoint'] = (data[from_col] + data[to_col]) / 2
    
    data = data.sort_values(by='Interval Midpoint')
    
    id_vars = [holeid_col, from_col, to_col, 'Interval Midpoint'] + hover_data_options
    melted_data = data.melt(id_vars=id_vars, value_vars=selected_analytes, var_name='Analyte', value_name='Result')

    melted_data['From'] = melted_data[from_col]
    melted_data['To'] = melted_data[to_col]

    unique_holeids = data[holeid_col].nunique()

    select_range = st.checkbox("Single-hole Interval Analyser", disabled=unique_holeids > 1)

    downholeplot = px.line(melted_data, x='Result', y='Interval Midpoint', color=holeid_col, line_group=holeid_col, markers=True, facet_col='Analyte', facet_col_wrap=4, hover_data=hover_data_options + ['From', 'To'])
    
    # Ensure Y-axis is reversed so that the top of the plot is the surface
    downholeplot.update_yaxes(autorange='reversed')

    downholeplot.update_layout(xaxis_title='Results', yaxis_title='Interval Midpoint', height=1000, title='Results by Drill Hole and Interval Midpoint')
    
    for i, facet in enumerate(downholeplot.select_traces()):
        downholeplot.update_xaxes(matches=None, row=1, col=i+1)

    if select_range and unique_holeids == 1:
        available_values = sorted(set(data[from_col].dropna()).union(set(data[to_col].dropna())))

        from_val = st.select_slider("Select the 'From' value", options=available_values, value=available_values[0], key="from_val")
        to_val = st.select_slider("Select the 'To' value", options=available_values, value=available_values[-1], key="to_val")

        st.write(f"Selected 'From' value: {from_val}")
        st.write(f"Selected 'To' value: {to_val}")

        downholeplot.update_traces(marker=dict(color=melted_data['Interval Midpoint'].apply(lambda x: 'red' if from_val <= x <= to_val else 'blue')))
    
        filtered_data = data[(data[from_col] >= from_val) & (data[to_col] <= to_val)]

        selected_analytes_for_avg = st.multiselect("Select analytes to calculate weighted averages", options=selected_analytes)
        
        if selected_analytes_for_avg:
            weighted_averages = []

            for analyte in selected_analytes_for_avg:
                analyte_data = filtered_data[[from_col, to_col, analyte]]
                analyte_data['Weight'] = analyte_data[to_col] - analyte_data[from_col]
                
                weighted_avg = (analyte_data[analyte] * analyte_data['Weight']).sum() / analyte_data['Weight'].sum()
                weighted_averages.append({'Analyte': analyte, 'Weighted Average': weighted_avg})

            weighted_avg_df = pd.DataFrame(weighted_averages)

            st.subheader("Weighted Averages for Selected Analytes")
            st.write(weighted_avg_df)

    st.plotly_chart(downholeplot, key="downholeplot")

# Calculcate unique combinations of selected variables, as well as occurences with an ability to plot with respect to a value average
def variabilityanalysis(data, holeid_col, from_col, to_col):
    groupby_columns = st.multiselect("Select columns to group by", options=data.columns)
    value_column = st.selectbox("Select value column to average", options=data.columns)

    if not groupby_columns or not value_column:
        return pd.DataFrame(columns=['Combination', 'Count', 'Counts_Percentage', 'Mean Value', 'Median Value', 'Min Value', 'Max Value', 'Range'])

    data[value_column] = pd.to_numeric(data[value_column], errors='coerce')

    data = data.dropna(subset=[value_column])

    data['unique_id'] = data[holeid_col].astype(str) + '_' + data[from_col].astype(str) + '_' + data[to_col].astype(str)
    combinations = data.groupby(groupby_columns)['unique_id'].nunique().reset_index()
    combinations = combinations.rename(columns={'unique_id': 'Count'})
    combinations['Combination'] = combinations[groupby_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    combinations["Counts_Percentage"] = (combinations["Count"] / combinations["Count"].sum()) * 100
    combinations["Mean Value"] = data.groupby(groupby_columns)[value_column].mean().values
    combinations["Median Value"] = data.groupby(groupby_columns)[value_column].median().values
    combinations["Min Value"] = data.groupby(groupby_columns)[value_column].min().values
    combinations["Max Value"] = data.groupby(groupby_columns)[value_column].max().values
    combinations["Range"] = combinations["Max Value"] - combinations["Min Value"]

    fig = px.bar(combinations, x='Combination', y='Mean Value', title=f'Mean {value_column} value with respect to {groupby_columns}', color='Counts_Percentage', color_continuous_scale='Viridis')
    fig2 = px.bar(combinations, x='Combination', y='Median Value', title=f'Median {value_column} value with respect to {groupby_columns}', color='Counts_Percentage', color_continuous_scale='Viridis')
    st.plotly_chart(fig, key="variabilityplot")
    st.plotly_chart(fig2, key="variabilityplot2")
    st.write(combinations)
    
    return combinations

# Function that allows the user to plot the weighted average of a parameter of interest for a given intervals/drillhole/dataset
def calculate_weighted_averages(data, holeid_col, from_col, to_col, holeid, from_val, to_val, additional_cols):
    
    matching_rows = data[(data[holeid_col] == holeid) & 
                         (data[from_col] >= from_val) & 
                         (data[to_col] <= to_val)]
    
    weighted_averages = {'HoleID': holeid, 'From': from_val, 'To': to_val}
    
    for additional_col in additional_cols:
        if additional_col in data.columns:
            weighted_avg = sum(matching_rows[additional_col] * matching_rows['Interval_Length']) / matching_rows['Interval_Length'].sum()
            weighted_averages[additional_col + '_Weighted_Avg'] = weighted_avg
    
    return weighted_averages

# Function that allows the user to investigate potential samples that meet mass, grade and categorical variable interests
def sampleselectionassistant(data, holeid_col, from_col, to_col):
    settings1, settings2 = st.columns([1, 1])
    with settings1:
        categorical_cols = st.multiselect("Select categorical variables for filtering (i.e., your subset for analysis)", options=data.columns)

        unique_values = {}
        for cat_col in categorical_cols:
            unique_values[cat_col] = data[cat_col].unique()

        categorical_vals = {}
        for cat_col in categorical_cols:
            categorical_vals[cat_col] = st.multiselect(f"Select categorical values for {cat_col} filtering", options=unique_values[cat_col])
    
    with settings2:
        parameter_col = st.selectbox("Select parameter to analyse (e.g., Cu_pct, K_pct, CuCN etc.)", options=data.columns)
        target_value = st.number_input(f"Enter target value for {parameter_col}", min_value=0.0)

        percentage_range = st.number_input("Enter allowable deviation as a percentage of target value", min_value=0.0, max_value=1000.0)

        required_mass = st.number_input("Enter required mass (unit agnostic)", min_value=0.0)  # Mass input field is now required
        mass_per_unit = st.number_input("Enter mass per unit of length (units = To - From)", min_value=0.0)

        select_all_holeid = st.checkbox("Select all Drillholes", value=True)
        if select_all_holeid:
            selected_drillholes = data[holeid_col].unique()
        else:
            selected_drillholes = st.multiselect("Select Drillholes", options=data[holeid_col].unique())

        filtered_data = data[data[holeid_col].isin(selected_drillholes)]
        
        actionbutton = st.button("Go!", key="actionbutton")

    if actionbutton:
        for cat_col in categorical_cols:
            if categorical_vals[cat_col]:
                filtered_data = filtered_data[filtered_data[cat_col].isin(categorical_vals[cat_col])]

        representative_intervals = filtered_data.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)

        representative_intervals['Interval_Length'] = representative_intervals[to_col] - representative_intervals[from_col]
        representative_intervals['Interval_Length'] = pd.to_numeric(representative_intervals['Interval_Length'], errors='coerce')
        representative_intervals['Mass'] = representative_intervals['Interval_Length'] * mass_per_unit
        representative_intervals['Mass'] = pd.to_numeric(representative_intervals['Mass'], errors='coerce')

        composite_intervals = []

        for hole_id in representative_intervals[holeid_col].unique():
            hole_data = representative_intervals[representative_intervals[holeid_col] == hole_id]

            # Create a sliding window over the intervals
            for start_idx in range(len(hole_data)):
                current_composite = []
                current_mass = 0
                weighted_sum = 0
                total_length = 0
                interval_start = hole_data.iloc[start_idx][from_col]

                # Expand the window to include consecutive intervals
                for end_idx in range(start_idx, len(hole_data)):
                    row = hole_data.iloc[end_idx]
                    current_composite.append(row)
                    current_mass += row['Mass']
                    weighted_sum += row[parameter_col] * row['Interval_Length']
                    total_length += row['Interval_Length']

                    if current_mass >= required_mass:
                        weighted_avg_parameter_value = weighted_sum / total_length
                        interval_end = row[to_col]

                        # Calculate how close the weighted average is to the target value
                        difference_from_target = abs(weighted_avg_parameter_value - target_value)

                        composite_intervals.append({
                            'HoleID': hole_id,
                            'From': interval_start,
                            'To': interval_end,
                            'Total_Mass': current_mass,
                            'Wt_Av_Parameter': weighted_avg_parameter_value,
                            'Target_Diff': difference_from_target
                        })

        composite_df = pd.DataFrame(composite_intervals)

        if composite_df.empty:
            st.text("### No valid composites could be created.")
            composite_df = pd.DataFrame()

        if not composite_df.empty:
            composite_df = composite_df.sort_values(by='Target_Diff')

            lower_bound = target_value - (target_value * (percentage_range / 100))
            upper_bound = target_value + (target_value * (percentage_range / 100))

            valid_composites = composite_df[(composite_df['Wt_Av_Parameter'] >= lower_bound) & 
                                            (composite_df['Wt_Av_Parameter'] <= upper_bound) & 
                                            (composite_df['Total_Mass'] >= required_mass)]
            st.write("### Proposed Samples")
            st.write(valid_composites)

        matching_intervals = []

        for _, composite in valid_composites.iterrows():
            holeid = composite['HoleID']
            from_val = composite['From']
            to_val = composite['To']

            matching_rows = data[(data[holeid_col] == holeid) & 
                                 (data[from_col] >= from_val) & 
                                 (data[to_col] <= to_val)]

            matching_intervals.append(matching_rows)

        filtered_intervals_df = pd.concat(matching_intervals)

        st.write("### Original Data for Valid Composite Intervals")
        st.write(filtered_intervals_df)

        return filtered_intervals_df

# Create a weighted average calculator        
def weighted_average(data, from_col, to_col):
    group_columns = st.multiselect("Select grouping variables (e.g., lithology, alteration)", options=data.columns, key="groupingvars")
    analyte_columns = st.multiselect("Select analytes for weighted average calculation", options=data.columns, key="weightedavgs")
    
    if not group_columns or not analyte_columns:
        st.write("Please select both grouping variables and analytes.")
        return

    weighted_avgs = []

    for group_vals, group in data.groupby(group_columns):

        for analyte_column in analyte_columns:
            group['Interval_Length'] = group[to_col] - group[from_col]

            weighted_sum = (group[analyte_column] * group['Interval_Length']).sum()
            total_length = group['Interval_Length'].sum()

            if total_length != 0:
                weighted_avg = weighted_sum / total_length
            else:
                weighted_avg = np.nan

            weighted_avgs.append({
                'Group': '_'.join(map(str, group_vals)),
                'Analyte': analyte_column,
                'Weighted_Avg': weighted_avg
            })

    weighted_avg_df = pd.DataFrame(weighted_avgs)

    st.write("Weighted Averages for Selected Analytes:")
    st.dataframe(weighted_avg_df)

# Create a scatter plot based on variables of interest to user
def scatteranalysis(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, key="scatterx")
    y_variable = st.selectbox("Y-axis variable", options=data.columns, key="scattery")
    colour_selection = st.selectbox("Colour selection", options=data.columns)
    trend_value = "ols" if st.checkbox("Select for ordinary least squares trendline") else None
    scatterplot = px.scatter(data, x=x_variable, y=y_variable, trendline=trend_value, color=colour_selection, title=f"Scatter plot of {x_variable} vs {y_variable}")
    st.plotly_chart(scatterplot, key="scatterplot")

# Create a box plot based on variables of interest to user
def boxplot(data):
    x_variable = st.selectbox("X-axis variable", options=data.columns, key="boxx")
    y_variable = st.selectbox("Y-axis variable", options=data.columns, key="boxy")
    colour_selection = st.selectbox("Colour selection", options=data.columns, key="colourselectbox")
    userboxplot = px.box(data, x=x_variable, y=y_variable, title=f"Box plot of {x_variable} vs {y_variable}", color=colour_selection)
    st.plotly_chart(userboxplot, key="userboxplot")

# Plot the drillhole traces in 3D space
def threedplot(data):
    xcoordinate = st.selectbox("Select X Coordinate", options=data.columns, key="x_coordinate_3d")
    ycoordinate = st.selectbox("Select Y Coordinate", options=data.columns, key="y_coordinate_3d")
    zcoordinate = st.selectbox("Select Z Coordinate", options=data.columns, key="z_coordinate_3d")
    hover_data = st.multiselect("Select Hover Data", options=data.columns, key="hover_data_3d")
    inverse_z = st.checkbox("Invert Z-axis", key="inverse_z")
    colour_variable = st.selectbox("Select Colour Variable", options=data.columns, key="colour_variable_3d")

    fig = px.scatter_3d(data, x=xcoordinate, y=ycoordinate, z=zcoordinate, color=colour_variable, hover_data=hover_data)

    if inverse_z:
        fig.update_layout(
            title='Drillhole Collars in 3D Space',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Depth',
                zaxis=dict(autorange="reversed"),
            ),
            height=900,
        )
    else:
        fig.update_layout(
            title='Drillhole Collars in 3D Space',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Depth',
            ),
            height=900,
        )
    marker_size = st.slider("Marker Size", min_value=1, max_value=10, value=3, key="marker_size_3d")
    fig.update_traces(marker=dict(size=marker_size))

    st.plotly_chart(fig, key="fig3")

# Function to calculate statistics for distribution analysis
def statistics_function(data, from_col, to_col):
    groupby_columns = st.multiselect("Select categorical variables for grouping (e.g., lithology, alteration)", 
                                     options=data.columns, key="group_columns")
    numeric_column = st.selectbox("Select a numeric column for analysis", options=data.columns, key="numeric_column_1")

    if not groupby_columns or not numeric_column:
        st.write("Please select both categorical variables for grouping and a numeric column.")
        return

    user_value = st.number_input(f"Enter a value to compare in the distribution of {numeric_column}", value=0.0, key="user_value_1")

    go_button = st.button("Go!", key="go2")

    if go_button:
        data['Interval_Length'] = data[to_col] - data[from_col]

        if len(groupby_columns) == 1:
            grouped_combinations = [groupby_columns]
        else:
            grouped_combinations = []
            for i in range(2, len(groupby_columns) + 1):
                for j in range(len(groupby_columns) - i + 1):
                    grouped_combinations.append(groupby_columns[j:j + i])

        results = []

        for group_columns in grouped_combinations:
            grouped_data = data.groupby(list(group_columns))

            for group_vals, group in grouped_data:
                group = group[group['Interval_Length'] != 0]

                mean_value = group[numeric_column].mean()
                std_dev = group[numeric_column].std()
                z_score = (user_value - mean_value) / std_dev if std_dev != 0 else np.nan
                median_value = group[numeric_column].median()

                results.append({
                    'Group': '_'.join(map(str, group_vals)),
                    'Combination': ', '.join(group_columns),
                    'Mean': mean_value,
                    'Median': median_value,
                    'Standard Deviation': std_dev,
                    'Z-score': z_score
                })

        results_df = pd.DataFrame(results)
        plot_box_plots_by_group(data, grouped_combinations, numeric_column, user_value, from_col, to_col)

        st.write("Calculated Statistics by Group:")  
        st.dataframe(results_df)

# Function to plot box plots by group
def plot_box_plots_by_group(data, group_combinations, numeric_column, user_value, from_col, to_col):
    data['Interval_Length'] = data[to_col] - data[from_col]
    
    for group_columns in group_combinations:
        grouped_data = data.groupby(list(group_columns))

        for group_vals, group in grouped_data:
            group_name = '_'.join(map(str, group_vals))
            try:
                fig_box = px.box(group, y=numeric_column, title=f"Boxplot of {numeric_column} for Group: {group_name}", points="all")
                fig_box.add_hline(y=user_value, line=dict(color="red", width=3, dash="dash"), annotation_text=f"Entered Value: {user_value}", annotation_position="top right")
                st.plotly_chart(fig_box)
            except Exception as e:
                st.error(f"Error generating plot for group {group_name}: {str(e)}")

# Function to merge two datasets based on overlapping intervals
def merge_on_overlap():
    st.write("### Upload File 1")
    file1 = st.file_uploader("Upload CSV File 1", type="csv", key="file1")
    st.write("### Upload File 2")
    file2 = st.file_uploader("Upload CSV File 2", type="csv", key="file2")

    if file1 is not None and file2 is not None:
        df1 = loaddata(file1)
        df2 = loaddata(file2)

    holeid_col_file1 = st.selectbox("Select 'holeid' column for File 1", options=df1.columns)
    holeid_col_file2 = st.selectbox("Select 'holeid' column for File 2", options=df2.columns)
    from_col_file1 = st.selectbox("Select 'from' column for File 1", options=df1.columns)
    to_col_file1 = st.selectbox("Select 'to' column for File 1", options=df1.columns)
    from_col_file2 = st.selectbox("Select 'from' column for File 2", options=df2.columns)
    to_col_file2 = st.selectbox("Select 'to' column for File 2", options=df2.columns)
    merge_col_file2 = st.selectbox("Select column to merge from File 2 to File 1", options=df2.columns)

    df1[from_col_file1] = pd.to_numeric(df1[from_col_file1], errors='coerce')
    df1[to_col_file1] = pd.to_numeric(df1[to_col_file1], errors='coerce')
    df2[from_col_file2] = pd.to_numeric(df2[from_col_file2], errors='coerce')
    df2[to_col_file2] = pd.to_numeric(df2[to_col_file2], errors='coerce')

    merge_button = st.button("Merge Data", key="merge_button")

    if merge_button:
        with st.spinner("Merging data... If this is a large dataset, you'll have to bear with me..."):

            df1[merge_col_file2] = None

            for index, row in df1.iterrows():
                holeid = row[holeid_col_file1]
                from_val = row[from_col_file1]
                to_val = row[to_col_file1]

                matching_rows_in_file2 = df2[df2[holeid_col_file2] == holeid]

                for _, file2_row in matching_rows_in_file2.iterrows():
                    file2_from = file2_row[from_col_file2]
                    file2_to = file2_row[to_col_file2]
                    file2_value = file2_row[merge_col_file2]

                    if not (to_val < file2_from or from_val > file2_to):
                        df1.at[index, merge_col_file2] = file2_value
                        break
            
            st.write("### Merged Data")
            st.write(df1)
    else:
        st.write("Please click the 'Merge Data' button to merge the datasets.")    
    return df1

# Defining the main execution function
def main():
    with st.sidebar:
        st.title("TerraVista 🌋")
        st.write("### Upload Data")
        uploaded_file = st.file_uploader("Upload your drillhole data file (CSV)", type=["csv"])
        drillholedata = loaddata(uploaded_file)
        if not drillholedata.empty:
            holeid_col = st.selectbox("Select your data's 'Drillhole ID' column", options=drillholedata.columns)
            from_col = st.selectbox("Select you data's 'From' column", options=drillholedata.columns)
            to_col = st.selectbox("Select your data's 'To' column", options=drillholedata.columns)
            st.write("### Filter Data (Prior to Analysis)")
            st.write("This is not a substitute for data cleaning. Please ensure your data is clean and formatted correctly. If you would like to use the data as is, select 'HoleID' (or equivalent), then 'Select All'.")

            selectedvariables = selectvariables(drillholedata)

            if selectedvariables:
                user_filtered_data = filterdata(selectedvariables, drillholedata)
            else:
                user_filtered_data = pd.DataFrame()
                st.text("Data will appear once selected")
        else:
            selectedvariables = []
            user_filtered_data = pd.DataFrame()
    
    if not selectedvariables:
        st.warning("Please upload a file and select at least one variable to filter on. If you want everything, select 'HoleID' (or equivalent), then 'Select All'. The purpose of filtering prior to displaying data and/or creating plots is that some datasets are enormous; and if not wishing to view everything - this saves you a lot of time!")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Downhole Plot", "Interval Variability Analysis", "Scatter Plot", "Box Plot", "3D Plot", "Sample Selection Assistant", "Weighted Average Calculator", "Distribution Analysis", "Merge Data"])
        
        if not user_filtered_data.empty:
            with tab1:
                st.header("Downhole Line Plot")
                try:
                    createdownholeplots(user_filtered_data, holeid_col, from_col, to_col)
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab2:
                try:
                    st.header("Interval Variability Analysis")
                    variabilityanalyses = variabilityanalysis(user_filtered_data, holeid_col, from_col, to_col)
                    if variabilityanalyses:
                        st.write(f"Number of Intervals Remaining: {variabilityanalyses['Count'].sum()}")
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab3:
                try:
                    st.header("Scatter Analysis")
                    scatteranalysis(user_filtered_data.reset_index(drop=True))
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab4:
                try:
                    st.header("Box Plot")
                    boxplot(user_filtered_data)
                except Exception as e:
                    with st.expander("Error Log", expanded=False):
                        st.error(f"An error occurred: {e}")
            with tab5:
                activate3d = st.radio("Activatation Status", ("Activated", "Deactivated"), horizontal=True, key="activate3d")
                if activate3d == "Activated":
                    try:
                        st.header("3D Plot")
                        threedplot(user_filtered_data)
                    except Exception as e:
                        with st.expander("Error Log", expanded=False):
                            st.error(f"An error occurred: {e}")
            with tab6:
                activatesa = st.radio("Activation Status", ("Activated", "Deactivated"), horizontal=True, key="activatesa")
                if activatesa == "Activated":
                    try:
                        st.header("Sample Selection Assistant")
                        sampleselectionassistant(user_filtered_data, holeid_col, from_col, to_col)   
                    except Exception as e:
                        with st.expander("Error Log", expanded=True):
                            st.error(f"An error occurred: {e}")
                            st.text("Detailed Traceback:")
                            st.text(traceback.format_exc())
            with tab7:
                try:
                    st.header("Weighted Average Calculator")
                    weighted_average(user_filtered_data, from_col, to_col)
                except Exception as e:
                    with st.expander("Error Log", expanded=True):
                        st.error(f"An error occurred: {e}")
            with tab8:
                try:
                    st.header("Distribution Analysis")
                    statistics_function(user_filtered_data, from_col, to_col)
                except Exception as e:
                    with st.expander("Error Log", expanded=True):
                        st.error(f"An error occurred: {e}")
            with tab9:
                try:
                    # The intent here is to merge two datasets based on overlapping intervals, which one can subsequently download as a CSV and reupload for further analysis
                    merge_on_overlap()
                except Exception as e:
                    with st.expander("Error Log", expanded=True):
                        st.error(f"An error occurred: {e}")

    with st.expander("Show Filtered Data"):
        st.header("Filtered Data Display")
        st.write(user_filtered_data)
      
    with st.expander("Help"):
        st.write("""
        ## How to Use This Application

        This application is designed to help you analyze drillhole data through various visualizations and analyses. Below is a detailed guide on how to use it:

        ### Step-by-Step Guide:
        1. **Upload Data**:
            - Use the sidebar to upload your drillhole data file.
            - The file must be in CSV format and encoded in UTF-8, Latin-1, or ISO-8859-1.
        2. **Filter Data**:
            - Select the variables you want to filter on and apply the desired filters.
            - Filters will be applied to all subsequent analyses.
        3. **Select Analysis**:
            - Choose the type of analysis you want to perform:
                - **Downhole Line Plot**: Visualize data along/down the drillhole.
                - **Interval Variability Analysis**: Analyze variability of intervals based on parameters like lithology or alteration types.
                - **Scatter Analysis**: Create scatter plots to explore relationships between variables.
                - **Box Plot**: Visualize the distribution of variables using box plots.
                - **3D Plot**: Plot drillhole collars in 3D space.
                - **Sample Selection Assistant**: Select samples based on criteria like mass requirements and cutoff grades.
                - **Weighted Average Calculator**: Calculate weighted averages for selected analytes.
                - **Distribution Analysis**: Analyze statistical distributions of numeric variables.
                - **Merge Data**: Merge two datasets based on overlapping intervals.

        ### Features:
        - **Data Upload**:
            - Supports CSV files with UTF-8, Latin-1, or ISO-8859-1 encoding.
            - Displays uploaded data for review.
        - **Filtering**:
            - Filter data based on values, ranges, or string matching.
            - Supports multi-select and "Select All" options for convenience.
        - **Visualization**:
            - Generate downhole line plots, scatter plots, box plots, and 3D plots.
            - Interactive plots with hover data and customizable options.
        - **Analysis**:
            - Perform interval variability analysis.
            - Calculate weighted averages for selected analytes.
            - Analyze statistical distributions with grouping options.
        - **Sample Selection**:
            - Assist in selecting samples based on mass, grade, and categorical variables.
            - Generate weighted averages for selected intervals.
        - **Data Merging**:
            - Merge two datasets based on overlapping intervals.

        ## Limitations:
        - **File Format**:
            - Only supports CSV files. Other formats like Excel or JSON are not supported.
        - **Data Quality**:
            - Assumes clean and properly formatted data. Missing or non-numeric values in critical columns may cause errors.
        - **Advanced Analysis**:
            - Does not support advanced statistical analyses beyond the provided tools.
        - **Performance**:
            - Large datasets may cause performance issues, especially for complex analyses or visualizations.

        ## Caveats:
        - **Filtering**:
            - Applying overly restrictive filters may result in no data being displayed.
        - **Encoding Issues**:
            - Ensure the file encoding matches one of the supported formats (UTF-8, Latin-1, ISO-8859-1).
        - **Numeric Columns**:
            - Ensure numeric columns are properly formatted. Non-numeric values will be coerced to NaN and may be dropped.
        - **3D Plotting**:
            - The Z-axis can be inverted for depth-based visualizations, but ensure the data is consistent.
        - **Sample Selection**:
            - The assistant assumes consistent units for mass and interval lengths.

        ## Troubleshooting:
        - **File Upload Issues**:
            - Ensure the file is in CSV format and encoded correctly.
        - **Data Not Displayed**:
            - Check if filters are too restrictive or if the data contains missing values.
        - **Errors in Analysis**:
            - Review the error log for details. Ensure the selected columns contain valid data.

        ## Feedback:
        If you encounter any issues or have suggestions for improvement, please contact refer to the error log for debugging, or worst case - reach out to a member of the Bundoora Technology Development Centre Geometallurgy Team.
        """)

# Having script execute as per convention
if __name__ == "__main__":
    main()
