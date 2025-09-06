import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns  
import yfinance as yf  
from datetime import datetime, timedelta 
import warnings  
warnings.filterwarnings('ignore') 

plt.style.use('seaborn-v0_8') 
sns.set_palette("husl") 

print("=== SECTOR PERFORMANCE COMPARISON PROJECT ===")
print("Analyzing performance across different sectors\n")

sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Finance": ["JPM", "BAC", "C", "GS", "MS"],
    "Energy": ["XOM", "CVX", "BP", "COP", "SLB"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
    "Consumer": ["PG", "KO", "PEP", "WMT", "COST"]
}

end_date = datetime.now()  
start_date = end_date - timedelta(days=365) 
print(f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

print("\n--- STEP 1: DOWNLOADING STOCK DATA ---")
all_stock_data = {} 
sector_stocks = []  

for sector, stocks in sectors.items():
    sector_stocks.extend(stocks) 
print(f"Downloading data for {len(sector_stocks)} stocks...")

try:
    stock_data = yf.download(sector_stocks, start=start_date, end=end_date)    
    if isinstance(stock_data.columns, pd.MultiIndex):
        if 'Adj Close' in stock_data.columns.levels[0]:
            stock_data = stock_data.xs('Adj Close', axis=1, level=0)
        elif 'Close' in stock_data.columns.levels[0]:
            stock_data = stock_data.xs('Close', axis=1, level=0)
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data")
    else:
        if 'Adj Close' in stock_data.columns:
            stock_data = stock_data[['Adj Close']]
        elif 'Close' in stock_data.columns:
            stock_data = stock_data[['Close']]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data")
    print("OK Data downloaded successfully!")
except Exception as e:
    print(f"X Error downloading data: {e}")
    exit()

print("\n--- STEP 2: CALCULATING DAILY RETURNS ---")
daily_returns = stock_data.pct_change().dropna() 
print(f" Calculated daily returns for {len(daily_returns.columns)} stocks")

print("\n--- STEP 3: CALCULATING SECTOR PERFORMANCE ---")
sector_returns = {} 
sector_daily_returns = {} 

for sector, stocks in sectors.items():   
    available_stocks = [stock for stock in stocks if stock in daily_returns.columns]
    if available_stocks:
        sector_daily_returns[sector] = daily_returns[available_stocks].mean(axis=1) 
        cumulative_returns = (1 + sector_daily_returns[sector]).cumprod() - 1 
        sector_returns[sector] = {
            'Total_Return': cumulative_returns.iloc[-1] * 100, 
            'Daily_Avg_Return': sector_daily_returns[sector].mean() * 100, 
            'Volatility': sector_daily_returns[sector].std() * 100, 
            'Sharpe_Ratio': (sector_daily_returns[sector].mean() / sector_daily_returns[sector].std()) * np.sqrt(252), 
            'Max_Drawdown': ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min() * 100, 
            'Stocks_Count': len(available_stocks)  
        }
        print(f" {sector}: {len(available_stocks)} stocks analyzed")

print("\n--- STEP 4: CREATING PERFORMANCE SUMMARY ---")
performance_df = pd.DataFrame(sector_returns).T  
performance_df = performance_df.round(2) 
print(" Performance summary created")

print("\n=== SECTOR PERFORMANCE SUMMARY ===")
print(performance_df.to_string())  

print("\n--- STEP 5: IDENTIFYING TOP PERFORMERS ---")
best_return = performance_df['Total_Return'].idxmax()  
worst_return = performance_df['Total_Return'].idxmin()  
best_sharpe = performance_df['Sharpe_Ratio'].idxmax()  
lowest_volatility = performance_df['Volatility'].idxmin() 

print(f" Best Total Return: {best_return} ({performance_df.loc[best_return, 'Total_Return']:.2f}%)")
print(f" Worst Total Return: {worst_return} ({performance_df.loc[worst_return, 'Total_Return']:.2f}%)")
print(f" Best Sharpe Ratio: {best_sharpe} ({performance_df.loc[best_sharpe, 'Sharpe_Ratio']:.2f})")
print(f" Lowest Volatility: {lowest_volatility} ({performance_df.loc[lowest_volatility, 'Volatility']:.2f}%)")

print("\n--- STEP 6: CREATING VISUALIZATIONS ---")

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
axes1[0].bar(performance_df.index, performance_df['Total_Return'], 
             color=sns.color_palette("husl", len(performance_df)))
axes1[0].set_title("Total Returns by Sector (%)")
axes1[0].set_ylabel("Total Return (%)")
axes1[0].tick_params(axis="x", rotation=45)

axes1[1].scatter(performance_df['Volatility'], performance_df['Total_Return'], 
                 s=100, alpha=0.7, c=range(len(performance_df)), cmap='viridis')
axes1[1].set_title("Risk vs Return Profile")
axes1[1].set_xlabel("Volatility (Risk) %")
axes1[1].set_ylabel("Total Return %")

for i, sector in enumerate(performance_df.index):
    axes1[1].annotate(sector,
                      (performance_df.loc[sector, 'Volatility'], performance_df.loc[sector, 'Total_Return']),
                      xytext=(5, 5), textcoords="offset points", fontsize=9)

plt.tight_layout()
plt.show()
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

axes2[0].barh(performance_df.index, performance_df['Sharpe_Ratio'],
              color=sns.color_palette("coolwarm", len(performance_df)))
axes2[0].set_title("Sharpe Ratio (Risk-Adjusted Return)")
axes2[0].set_xlabel("Sharpe Ratio")

for i, v in enumerate(performance_df['Sharpe_Ratio']):
    axes2[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontweight="bold")

for sector in sector_daily_returns.keys():
    cumulative = (1 + sector_daily_returns[sector]).cumprod() - 1
    axes2[1].plot(cumulative.index, cumulative * 100, label=sector, linewidth=2)

axes2[1].set_title("Cumulative Returns Comparison Over Time")
axes2[1].set_ylabel("Cumulative Return (%)")
axes2[1].set_xlabel("Date")
axes2[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
axes2[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("\n--- STEP 7: SECTOR CORRELATION ANALYSIS ---")

sector_returns_df = pd.DataFrame(sector_daily_returns) 
correlation_matrix = sector_returns_df.corr() 

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation'}) 
plt.title('Sector Correlation Matrix\n(How sectors move together)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n--- STEP 8: COMPREHENSIVE RANKING ---")

performance_df['Risk_Adjusted_Score'] = (performance_df['Total_Return'] / performance_df['Volatility']) 
performance_df['Overall_Rank'] = performance_df['Sharpe_Ratio'].rank(ascending=False) 

ranked_sectors = performance_df.sort_values('Overall_Rank') 

print(" SECTOR RANKINGS (Based on Risk-Adjusted Returns):")
for i, (sector, data) in enumerate(ranked_sectors.iterrows(), 1):
    print(f"{i}. {sector:15} | Return: {data['Total_Return']:6.2f}% | Risk: {data['Volatility']:5.2f}% | Sharpe: {data['Sharpe_Ratio']:5.2f}")

print("\n--- STEP 9: INVESTMENT INSIGHTS ---")
print(" INVESTMENT RECOMMENDATIONS:")
print(f"• GROWTH PICK: {best_return} - Highest absolute returns ({performance_df.loc[best_return, 'Total_Return']:.2f}%)")
print(f"• BALANCED PICK: {best_sharpe} - Best risk-adjusted returns (Sharpe: {performance_df.loc[best_sharpe, 'Sharpe_Ratio']:.2f})")
print(f"• DEFENSIVE PICK: {lowest_volatility} - Lowest volatility ({performance_df.loc[lowest_volatility, 'Volatility']:.2f}%)")

high_correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if corr_value > 0.7: 
            high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))

if high_correlation_pairs:
    print(f"\n  DIVERSIFICATION ALERT:")
    for sector1, sector2, corr in high_correlation_pairs:
        print(f"   {sector1} and {sector2} are highly correlated ({corr:.2f}) - consider diversifying")
else:
    print(f"\n DIVERSIFICATION: All sectors show good diversification potential")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ")
print("Key Takeaways:")
print("1. Compare total returns AND risk (volatility)")
print("2. Sharpe ratio shows risk-adjusted performance")
print("3. Correlation helps in portfolio diversification")
print("4. Past performance doesn't guarantee future results")
print("="*60)