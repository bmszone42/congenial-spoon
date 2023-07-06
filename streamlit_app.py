import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time
import base64

# Define a class to hold session state
class _SessionState:
    def __init__(self, config):
        self.config = config
        self.num_iterations = 100
        self.min_increase = -0.1
        self.max_increase = 0.1
        self.cost_penalty = 1000

# Function to simulate cost and throughput over time
def simulate(state, option):
    # Initial cost and throughput
    cost = state.config[option]['cost']
    throughput = state.config[option]['throughput']

    # Generate random increases in cost and throughput
    cost_increases = np.random.uniform(state.min_increase, state.max_increase, state.num_iterations) * cost
    throughput_increases = np.random.uniform(state.min_increase, state.max_increase, state.num_iterations) * throughput

    # Calculate cumulative cost and throughput
    costs = np.cumsum(np.append(cost, cost_increases))
    throughputs = np.cumsum(np.append(throughput, throughput_increases))

    # Add cost penalty for high throughput increase
    costs[throughput_increases > 0.1] += state.cost_penalty

    return costs, throughputs

# Function to plot data
def plot_fig(df, title):
    fig = go.Figure()
    colors = ['blue', 'red', 'green']
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(y=df[col], mode='lines', name=col, line=dict(color=colors[i])))
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)

# Function to handle sidebar inputs
def handle_sidebar(state):
    st.sidebar.title('ASIC/FPGA/Chiplets Simulator')
    options_to_simulate = st.sidebar.multiselect('Options to simulate', list(state.config.keys()), default=list(state.config.keys()))
    for option in options_to_simulate:
        state.config[option]['cost'] = st.sidebar.number_input(f'Initial {option} cost', min_value=500, max_value=100000, value=state.config[option]['cost'])
        state.config[option]['throughput'] = st.sidebar.number_input(f'Initial {option} throughput', min_value=1, max_value=5000, value=state.config[option]['throughput'])
    state.num_iterations = st.sidebar.slider("Number of iterations", 1, 1000, state.num_iterations)
    state.min_increase = st.sidebar.slider("Minimum percentage increase", -1.0, 1.0, state.min_increase)
    state.max_increase = st.sidebar.slider("Maximum percentage increase", -1.0, 1.0, state.max_increase)
    state.cost_penalty = st.sidebar.number_input("Cost penalty for high throughput increase", min_value=0, max_value=10000, value=state.cost_penalty)

    if state.max_increase < state.min_increase:
        st.sidebar.error("Maximum percentage increase must be greater than or equal to minimum percentage increase")

    new_option = st.sidebar.text_input("Add a new option to simulate")
    if new_option:
        state.config[new_option] = {'cost': 10000, 'throughput': 1000}

    if st.sidebar.button("Reset"):
        state = _SessionState(config)

    return options_to_simulate

# Function to create a download link
def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some
    # strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Main function
def main():
    # Initial configuration for ASIC, FPGA, and Chiplets
    config = {
        'ASIC': {'cost': 10000, 'throughput': 1000},
        'FPGA': {'cost': 5000, 'throughput': 500},
        'Chiplets': {'cost': 7500, 'throughput': 750}
    }
    state = _SessionState(config)

    # Handle sidebar inputs
    options_to_simulate = handle_sidebar(state)

    # Run simulation button
    if st.button("Run Simulation"):
        # Start timer for performance monitoring
        start_time = time.time()

        # Simulate costs and throughputs
        costs = {}
        throughputs = {}
        for option in options_to_simulate:
            costs[option], throughputs[option] = simulate(state, option)

        # Create dataframes for costs and throughputs
        cost_df = pd.DataFrame({option: costs[option] for option in options_to_simulate})
        throughput_df = pd.DataFrame({option: throughputs[option] for option in options_to_simulate})

        # Display simulation results
        st.title('Simulation Results')
        plot_fig(cost_df, 'Cost over iterations')
        plot_fig(throughput_df, 'Throughput over iterations')

        # Add a histogram of costs and throughputs
        st.title('Histogram of Costs and Throughputs')
        fig = go.Figure()
        for option in options_to_simulate:
            fig.add_trace(go.Histogram(x=costs[option], name=f'{option} Cost'))
            fig.add_trace(go.Histogram(x=throughputs[option], name=f'{option} Throughput'))
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig)

        # Display final statistics
        st.header('Simulation Statistics')
        summary_df = pd.DataFrame({option: [costs[option][-1], throughputs[option][-1], np.mean(costs[option]), np.mean(throughputs[option]), np.std(costs[option]), np.std(throughputs[option])] for option in options_to_simulate}, index=['Final Cost', 'Final Throughput', 'Mean Cost', 'Mean Throughput', 'Cost Standard Deviation', 'Throughput Standard Deviation'])
        st.table(summary_df)

        # Save results button
        if st.button("Save results"):
            # Concatenate cost and throughput dataframes
            result_df = pd.concat([cost_df, throughput_df], axis=1)
            # Create download link and display it
            download_link = create_download_link(result_df, "simulation_results.csv")
            st.markdown(download_link, unsafe_allow_html=True)

        # End timer and print performance
        end_time = time.time()
        st.write("Performance: %s seconds" % (end_time - start_time))

# Run the app
if __name__ == "__main__":
    main()
