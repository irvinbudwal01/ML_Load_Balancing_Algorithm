# data_loader.py
import pandas as pd

def load_network_data():
    data = {
        "packets_dropped": [0.05, 0.02, 0.07, 0.03, 0.04, 0.06, 0.01, 0.02, 0.08, 0.03, 0.04, 0.05, 0.02, 0.07, 0.06, 0.03, 0.04, 0.05, 0.01, 0.02],
        # Deviation in server load (lower values mean more balanced usage)
        "server_utilization": [0.1, 0.05, 0.12, 0.08, 0.06, 0.1, 0.03, 0.04, 0.11, 0.07, 0.08, 0.1, 0.05, 0.12, 0.09, 0.07, 0.06, 0.1, 0.02, 0.03],   
        "latency": [20, 25, 15, 30, 22, 18, 26, 23, 19, 24, 21, 27, 16, 29, 20, 19, 22, 25, 18, 17]
    }
    # Each row row represents a time step

    df = pd.DataFrame(data)
    return df
