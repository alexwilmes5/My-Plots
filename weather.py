import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set page configuration
st.set_page_config(page_title="Interactive Matplotlib Dashboard", layout="wide")

# Title
st.title("🎯 Interactive Matplotlib Dashboard")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns(2)

# Plot 1: Interactive Sine/Cosine Wave
with col1:
    st.subheader("📈 Wave Functions")
    
    # Controls for wave plot
    wave_type = st.selectbox("Wave Type", ["Sine", "Cosine", "Both"], key="wave")
    frequency = st.slider("Frequency", 0.5, 5.0, 1.0, 0.1, key="freq")
    amplitude = st.slider("Amplitude", 0.5, 3.0, 1.0, 0.1, key="amp")
    phase = st.slider("Phase Shift", 0.0, 2*np.pi, 0.0, 0.1, key="phase")
    
    # Generate wave data
    x = np.linspace(0, 4*np.pi, 1000)
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    if wave_type == "Sine":
        y = amplitude * np.sin(frequency * x + phase)
        ax1.plot(x, y, 'b-', linewidth=2, label='Sine')
    elif wave_type == "Cosine":
        y = amplitude * np.cos(frequency * x + phase)
        ax1.plot(x, y, 'r-', linewidth=2, label='Cosine')
    else:  # Both
        y1 = amplitude * np.sin(frequency * x + phase)
        y2 = amplitude * np.cos(frequency * x + phase)
        ax1.plot(x, y1, 'b-', linewidth=2, label='Sine')
        ax1.plot(x, y2, 'r-', linewidth=2, label='Cosine')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'{wave_type} Wave (f={frequency}, A={amplitude})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-4, 4)
    
    st.pyplot(fig1)
    plt.close(fig1)

# Plot 2: Interactive Scatter Plot with Normal Distribution
with col2:
    st.subheader("🎲 Random Scatter Plot")
    
    # Controls for scatter plot
    n_points = st.slider("Number of Points", 50, 500, 100, 10, key="points")
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5, 0.1, key="noise")
    color_scheme = st.selectbox("Color Scheme", ["viridis", "plasma", "coolwarm", "spring"], key="colors")
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    x_scatter = np.random.normal(0, 1, n_points)
    y_scatter = 2 * x_scatter + np.random.normal(0, noise_level, n_points)
    colors = np.random.rand(n_points)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(x_scatter, y_scatter, c=colors, cmap=color_scheme, alpha=0.7, s=50)
    
    # Add trend line
    z = np.polyfit(x_scatter, y_scatter, 1)
    p = np.poly1d(z)
    ax2.plot(x_scatter, p(x_scatter), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('X Values')
    ax2.set_ylabel('Y Values')
    ax2.set_title(f'Scatter Plot ({n_points} points, noise={noise_level})')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2)
    
    st.pyplot(fig2)
    plt.close(fig2)

# Plot 3: Interactive Histogram
with col1:
    st.subheader("📊 Distribution Histogram")
    
    # Controls for histogram
    distribution = st.selectbox("Distribution Type", ["Normal", "Exponential", "Uniform"], key="dist")
    sample_size = st.slider("Sample Size", 100, 2000, 500, 50, key="sample")
    bins = st.slider("Number of Bins", 10, 100, 30, 5, key="bins")
    
    # Generate data based on distribution
    np.random.seed(123)
    if distribution == "Normal":
        data = np.random.normal(0, 1, sample_size)
        color = 'skyblue'
    elif distribution == "Exponential":
        data = np.random.exponential(1, sample_size)
        color = 'lightcoral'
    else:  # Uniform
        data = np.random.uniform(-3, 3, sample_size)
        color = 'lightgreen'
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    n, bins_edges, patches = ax3.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax3.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1 Std: {std_val:.2f}')
    ax3.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{distribution} Distribution (n={sample_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    st.pyplot(fig3)
    plt.close(fig3)

# Plot 4: Interactive Polar Plot
with col2:
    st.subheader("🌀 Polar Plot")
    
    # Controls for polar plot
    pattern = st.selectbox("Pattern Type", ["Rose", "Spiral", "Lemniscate"], key="pattern")
    petals = st.slider("Petals/Loops", 2, 8, 4, 1, key="petals")
    radius_scale = st.slider("Radius Scale", 0.5, 3.0, 1.0, 0.1, key="radius")
    
    # Generate polar data
    theta = np.linspace(0, 4*np.pi, 1000)
    
    fig4, ax4 = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
    
    if pattern == "Rose":
        r = radius_scale * np.cos(petals * theta)
        ax4.plot(theta, r, 'purple', linewidth=2)
        ax4.fill(theta, r, alpha=0.3, color='purple')
    elif pattern == "Spiral":
        r = radius_scale * theta / (2*np.pi)
        ax4.plot(theta, r, 'green', linewidth=2)
    else:  # Lemniscate
        r = radius_scale * np.sqrt(np.cos(2 * theta))
        # Handle complex numbers by taking only real part where valid
        r = np.where(np.cos(2 * theta) >= 0, r, np.nan)
        ax4.plot(theta, r, 'blue', linewidth=2)
        ax4.plot(theta, -r, 'blue', linewidth=2)
    
    ax4.set_title(f'{pattern} Pattern (n={petals})', pad=20)
    ax4.grid(True)
    
    st.pyplot(fig4)
    plt.close(fig4)

# Add some information at the bottom
st.markdown("---")
st.markdown("""
### 🎮 Interactive Controls
- **Wave Functions**: Adjust frequency, amplitude, and phase of trigonometric functions
- **Scatter Plot**: Control the number of points, noise level, and color scheme
- **Histogram**: Choose different statistical distributions and visualization parameters  
- **Polar Plot**: Create beautiful geometric patterns with adjustable parameters

*Built with Streamlit and Matplotlib*
""")