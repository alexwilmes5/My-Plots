"""
Comprehensive Weather Data Analysis
Author: [Alex Wilmes]
Description: Analyzing weather patterns using matplotlib for data visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherAnalyzer:
    def __init__(self):
        self.data = None
        self.city_name = "Sample City"
        
    def generate_sample_data(self, years=3):
        """Generate realistic sample weather data for demonstration"""
        print("Generating sample weather data...")
        
        # Create date range
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic weather patterns
        n_days = len(dates)
        
        # Temperature with seasonal variation
        day_of_year = dates.dayofyear
        base_temp = 15 + 20 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        temp_noise = np.random.normal(0, 5, n_days)
        temperature = base_temp + temp_noise
        
        # Humidity (inversely related to temperature with noise)
        humidity = 70 - 0.5 * (temperature - 15) + np.random.normal(0, 10, n_days)
        humidity = np.clip(humidity, 20, 95)
        
        # Precipitation (more likely in certain seasons)
        precip_prob = 0.3 + 0.2 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        precipitation = np.where(
            np.random.random(n_days) < precip_prob,
            np.random.exponential(5, n_days),
            0
        )
        
        # Wind speed
        wind_speed = np.random.gamma(2, 3, n_days)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': wind_speed
        })
        
        self.data['month'] = self.data['date'].dt.month
        self.data['season'] = self.data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        print(f"Generated {len(self.data)} days of weather data")
        return self.data
    
    def create_overview_dashboard(self):
        """Create a comprehensive dashboard of weather patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Weather Analysis Dashboard - {self.city_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Temperature trend over time
        axes[0, 0].plot(self.data['date'], self.data['temperature'], 
                       alpha=0.7, linewidth=0.8, color='red')
        axes[0, 0].set_title('Temperature Trend Over Time')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add trend line
        x_numeric = mdates.date2num(self.data['date'])
        z = np.polyfit(x_numeric, self.data['temperature'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.data['date'], p(x_numeric), 
                       "r--", alpha=0.8, linewidth=2, label='Trend')
        axes[0, 0].legend()
        
        # 2. Monthly temperature distribution
        monthly_temps = [self.data[self.data['month'] == i]['temperature'].values 
                        for i in range(1, 13)]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        box_plot = axes[0, 1].boxplot(monthly_temps, labels=months, patch_artist=True)
        axes[0, 1].set_title('Monthly Temperature Distribution')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.coolwarm(np.linspace(0, 1, 12))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # 3. Precipitation patterns
        monthly_precip = self.data.groupby('month')['precipitation'].sum()
        bars = axes[1, 0].bar(months, monthly_precip, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Total Monthly Precipitation')
        axes[1, 0].set_ylabel('Precipitation (mm)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
        
        # 4. Temperature vs Humidity correlation
        scatter = axes[1, 1].scatter(self.data['temperature'], self.data['humidity'], 
                                   alpha=0.6, c=self.data['precipitation'], 
                                   cmap='Blues', s=20)
        axes[1, 1].set_title('Temperature vs Humidity\n(Color = Precipitation)')
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Humidity (%)')
        
        # Add correlation line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.data['temperature'], self.data['humidity'])
        line = slope * self.data['temperature'] + intercept
        axes[1, 1].plot(self.data['temperature'], line, 'r--', alpha=0.8)
        axes[1, 1].text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.colorbar(scatter, ax=axes[1, 1], label='Precipitation (mm)')
        
        plt.tight_layout()
        plt.savefig('weather_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_seasonal_patterns(self):
        """Analyze and visualize seasonal weather patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Seasonal Weather Patterns Analysis', fontsize=16, fontweight='bold')
        
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        season_colors = {'Spring': 'green', 'Summer': 'orange', 
                        'Fall': 'brown', 'Winter': 'blue'}
        
        # 1. Seasonal temperature comparison
        for season in seasons:
            season_data = self.data[self.data['season'] == season]
            axes[0, 0].hist(season_data['temperature'], alpha=0.6, 
                           label=season, color=season_colors[season], bins=20)
        
        axes[0, 0].set_title('Temperature Distribution by Season')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Seasonal precipitation patterns
        seasonal_precip = self.data.groupby('season')[['precipitation', 'humidity']].mean()
        x_pos = np.arange(len(seasons))
        
        bars1 = axes[0, 1].bar(x_pos - 0.2, seasonal_precip['precipitation'], 
                              0.4, label='Precipitation (mm)', color='lightblue')
        ax2 = axes[0, 1].twinx()
        bars2 = ax2.bar(x_pos + 0.2, seasonal_precip['humidity'], 
                       0.4, label='Humidity (%)', color='lightcoral')
        
        axes[0, 1].set_title('Seasonal Precipitation & Humidity')
        axes[0, 1].set_xlabel('Seasons')
        axes[0, 1].set_ylabel('Precipitation (mm)')
        ax2.set_ylabel('Humidity (%)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(seasons)
        
        # Combined legend
        lines1, labels1 = axes[0, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. Wind patterns by season
        wind_data = [self.data[self.data['season'] == season]['wind_speed'].values 
                    for season in seasons]
        
        violin_parts = axes[1, 0].violinplot(wind_data, positions=range(1, 5))
        axes[1, 0].set_title('Wind Speed Distribution by Season')
        axes[1, 0].set_ylabel('Wind Speed (km/h)')
        axes[1, 0].set_xticks(range(1, 5))
        axes[1, 0].set_xticklabels(seasons)
        
        # Color the violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(list(season_colors.values())[i])
            pc.set_alpha(0.7)
        
        # 4. Extreme weather events
        # Define extremes
        temp_extremes = self.data[
            (self.data['temperature'] < self.data['temperature'].quantile(0.05)) |
            (self.data['temperature'] > self.data['temperature'].quantile(0.95))
        ]
        
        extreme_counts = temp_extremes.groupby('season').size()
        bars = axes[1, 1].bar(seasons, [extreme_counts.get(s, 0) for s in seasons],
                             color=[season_colors[s] for s in seasons], alpha=0.7)
        
        axes[1, 1].set_title('Extreme Temperature Events by Season')
        axes[1, 1].set_ylabel('Number of Extreme Days')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_correlation_matrix(self):
        """Create a correlation matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                             center=0, square=True, mask=mask, 
                             linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Weather Variables Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation insights
        print("\n=== Correlation Analysis ===")
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                print(f"{numeric_cols[i]} vs {numeric_cols[j]}: {corr_val:.3f}")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*50)
        print("WEATHER DATA ANALYSIS SUMMARY REPORT")
        print("="*50)
        
        print(f"\nDataset Overview:")
        print(f"• Analysis period: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}")
        print(f"• Total days analyzed: {len(self.data)}")
        print(f"• Location: {self.city_name}")
        
        print(f"\nTemperature Statistics:")
        print(f"• Average temperature: {self.data['temperature'].mean():.1f}°C")
        print(f"• Temperature range: {self.data['temperature'].min():.1f}°C to {self.data['temperature'].max():.1f}°C")
        print(f"• Hottest month: {self.data.groupby('month')['temperature'].mean().idxmax()}")
        print(f"• Coldest month: {self.data.groupby('month')['temperature'].mean().idxmin()}")
        
        print(f"\nPrecipitation Statistics:")
        print(f"• Total precipitation: {self.data['precipitation'].sum():.1f}mm")
        print(f"• Average daily precipitation: {self.data['precipitation'].mean():.1f}mm")
        print(f"• Rainy days: {(self.data['precipitation'] > 0).sum()} ({(self.data['precipitation'] > 0).mean()*100:.1f}%)")
        print(f"• Wettest month: {self.data.groupby('month')['precipitation'].sum().idxmax()}")
        
        print(f"\nOther Metrics:")
        print(f"• Average humidity: {self.data['humidity'].mean():.1f}%")
        print(f"• Average wind speed: {self.data['wind_speed'].mean():.1f} km/h")
        
        # Seasonal comparison
        print(f"\nSeasonal Averages:")
        seasonal_avg = self.data.groupby('season')[['temperature', 'humidity', 'precipitation']].mean()
        for season in seasonal_avg.index:
            print(f"• {season}: {seasonal_avg.loc[season, 'temperature']:.1f}°C, "
                  f"{seasonal_avg.loc[season, 'humidity']:.1f}% humidity, "
                  f"{seasonal_avg.loc[season, 'precipitation']:.1f}mm precip")

def main():
    """Main function to run the complete weather analysis"""
    print("Starting Comprehensive Weather Data Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = WeatherAnalyzer()
    
    # Generate sample data (replace this with real data loading if available)
    analyzer.generate_sample_data(years=3)
    
    # Run all analyses
    print("\n1. Creating overview dashboard...")
    analyzer.create_overview_dashboard()
    
    print("\n2. Analyzing seasonal patterns...")
    analyzer.analyze_seasonal_patterns()
    
    print("\n3. Creating correlation matrix...")
    analyzer.create_correlation_matrix()
    
    print("\n4. Generating summary report...")
    analyzer.generate_summary_report()
    
    print("\n" + "="*50)
    print("Analysis complete! Check the generated PNG files:")
    print("• weather_dashboard.png")
    print("• seasonal_analysis.png") 
    print("• correlation_matrix.png")
    print("="*50)

if __name__ == "__main__":
    main()