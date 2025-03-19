# visualization functions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_data(data_path='data/selected_features_data.csv'):
    data = pd.read_csv(data_path)
    readmission_counts = {
        '<30': data['readmitted_<30'].sum(),
        '>30': data['readmitted_>30'].sum(),
        'NO': data['readmitted_NO'].sum()
    }
    # plot
    plt.figure(figsize=(10, 6))
    colors = ['#FF4136', '#2ECC40', '#0074D9']
    plt.bar(readmission_counts.keys(), readmission_counts.values(), color=colors)
    plt.xlabel('Readmission Status')
    plt.ylabel('Count')
    plt.title('Readmission Counts')
    plt.show()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF4136', label='<30 days'),
        Patch(facecolor='#2ECC40', label='>30 days'),
        Patch(facecolor='#0074D9', label='NO readmission')
    ]
    for i, (category, count) in enumerate(readmission_counts.items()):
        plt.text(i, count + 100, f'{count}', ha='center', fontweight='bold')
    
    # save to directory
    output_dir='outputs'
    plt.savefig(os.path.join(output_dir, 'readmission_counts.png'))
    plt.show()

if __name__ == '__main__':
    plot_data()