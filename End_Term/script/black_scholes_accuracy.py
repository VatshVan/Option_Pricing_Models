import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import os
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIRECTORY = 'data/'
PLOTS_DIRECTORY = 'plots/'
NIFTY_SPOT_PRICE = 24363.30 # The data used is the options available at 8th August, 2025 15:30 PM
RISK_FREE_RATE = 0.069
ANALYSIS_DATE = datetime(2025, 8, 8)

def black_scholes_pricer(S, K, T, r, sigma, option_type):
    """Calculates the Black-Scholes option price."""
    if T <= 0 or sigma <= 0: return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.nan

def get_expiry_from_filename(filename):
    """Extracts the expiry date from a filename."""
    match = re.search(r'(\d{1,2}-[A-Za-z]{3}-\d{4})', filename)
    if match: return datetime.strptime(match.group(1), '%d-%b-%Y')
    return None

def parse_option_chain_file(filepath):
    """Parses the option chain file using the robust built-in csv module."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None); next(reader, None)
        for parts in reader:
            if len(parts) != 23: continue
            try:
                cleaned_parts = [p.strip().replace(',', '').replace('-', 'nan') for p in parts]
                strike_price_str = cleaned_parts[11]
                if float(cleaned_parts[5]) > 0 and float(cleaned_parts[4]) > 0:
                    records.append({'OptionType': 'Call', 'StrikePrice': float(strike_price_str), 'RealPrice': float(cleaned_parts[5]), 'IV': float(cleaned_parts[4])})
                if float(cleaned_parts[19]) > 0 and float(cleaned_parts[20]) > 0:
                     records.append({'OptionType': 'Put', 'StrikePrice': float(strike_price_str), 'RealPrice': float(cleaned_parts[17]), 'IV': float(cleaned_parts[18])})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(records)

os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

all_data_frames = []
for filename in os.listdir(DATA_DIRECTORY):
    if filename.endswith('.csv'):
        expiry_date = get_expiry_from_filename(filename)
        if not expiry_date:
            print(f"Warning: Could not determine expiry for {filename}")
            continue
        filepath = os.path.join(DATA_DIRECTORY, filename)
        df = parse_option_chain_file(filepath)
        if not df.empty:
            df['ExpiryDate'] = expiry_date
            all_data_frames.append(df)
        else:
            print(f"Warning: No valid option data found in {filename}.")

if not all_data_frames:
    print("\nError: No data was successfully processed.")
    exit()

master_df = pd.concat(all_data_frames, ignore_index=True)
master_df['TimeToExpiry'] = (master_df['ExpiryDate'] - ANALYSIS_DATE).dt.days / 365.25
master_df['UnderlyingPrice'] = NIFTY_SPOT_PRICE
master_df['Sigma'] = master_df['IV'] / 100.0
master_df['BS_Price'] = master_df.apply(lambda row: black_scholes_pricer(S=row['UnderlyingPrice'], K=row['StrikePrice'], T=row['TimeToExpiry'], r=RISK_FREE_RATE, sigma=row['Sigma'], option_type=row['OptionType']), axis=1)
master_df.dropna(subset=['BS_Price'], inplace=True)
master_df['Error'] = master_df['RealPrice'] - master_df['BS_Price']
master_df['AbsolutePctError'] = (abs(master_df['Error']) / master_df['RealPrice']) * 100

def get_moneyness(row):
    atm_threshold = 0.02
    if row['OptionType'] == 'Call':
        if row['StrikePrice'] < NIFTY_SPOT_PRICE * (1 - atm_threshold): return 'In-the-Money (ITM)'
        elif row['StrikePrice'] > NIFTY_SPOT_PRICE * (1 + atm_threshold): return 'Out-of-the-Money (OTM)'
        else: return 'At-the-Money (ATM)'
    else:
        if row['StrikePrice'] > NIFTY_SPOT_PRICE * (1 + atm_threshold): return 'In-the-Money (ITM)'
        elif row['StrikePrice'] < NIFTY_SPOT_PRICE * (1 - atm_threshold): return 'Out-of-the-Money (OTM)'
        else: return 'At-the-Money (ATM)'
master_df['Moneyness'] = master_df.apply(get_moneyness, axis=1)
expiry_bins = [0, 30/365.25, 90/365.25, np.inf]
expiry_labels = ['Near-Term (0-30 days)', 'Mid-Term (31-90 days)', 'Long-Term (>90 days)']
master_df['ExpiryBin'] = pd.cut(master_df['TimeToExpiry'], bins=expiry_bins, labels=expiry_labels, right=False)

def series_to_markdown_table(series, headers):
    """Converts a pandas Series to a Markdown table string."""
    header_str = f"| {headers[0]:<25} | {headers[1]:<10} |\n"
    line_str = f"|{'-'*27}|{'-'*12}|\n"
    table = header_str + line_str
    for index, value in series.items():
        table += f"| {str(index):<25} | {value:<10.2f} |\n"
    return table

overall_mape = master_df['AbsolutePctError'].mean()
error_by_type = master_df.groupby('OptionType')['AbsolutePctError'].mean()
error_by_moneyness = master_df.groupby('Moneyness')['AbsolutePctError'].mean().sort_values()
error_by_expiry = master_df.groupby('ExpiryBin', observed=True)['AbsolutePctError'].mean()

report_string = f"""
# Black-Scholes Model Accuracy Report

### Analysis Parameters
- **Analysis Date:** {ANALYSIS_DATE.strftime('%Y-%m-%d')}
- **Nifty Spot Price:** ₹{NIFTY_SPOT_PRICE:,.2f}
- **Risk-Free Rate:** {RISK_FREE_RATE * 100:.2f}%

---

### Overall Model Error
This value shows the average percentage difference between the model's price and the real market price.

- **Mean Absolute Percentage Error (MAPE): {overall_mape:.2f}%**

---

## Detailed Error Breakdown

### MAPE by Option Type
This table shows the model's average error for Call options vs. Put options.

{series_to_markdown_table(error_by_type, ['Option Type', 'MAPE (%)'])}
### MAPE by Moneyness
This table shows the model's average error based on whether an option is In-the-Money (ITM), At-the-Money (ATM), or Out-of-the-Money (OTM).

{series_to_markdown_table(error_by_moneyness, ['Moneyness', 'MAPE (%)'])}
### MAPE by Time to Expiry
This table shows the model's average error for options with different expiration timeframes.

{series_to_markdown_table(error_by_expiry, ['Expiry Bin', 'MAPE (%)'])}
"""

with open('script/README.md', 'w', encoding='utf-8') as f:
    f.write(report_string)

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(master_df[master_df['OptionType'] == 'Call']['RealPrice'], master_df[master_df['OptionType'] == 'Call']['BS_Price'], alpha=0.6, label='Calls')
ax.scatter(master_df[master_df['OptionType'] == 'Put']['RealPrice'], master_df[master_df['OptionType'] == 'Put']['BS_Price'], alpha=0.6, label='Puts')
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Match (y=x)')
ax.set_xlabel('Real Market Price (₹)'); ax.set_ylabel('Black-Scholes Price (₹)')
ax.set_title('Model Accuracy: Real Price vs. Black-Scholes Price')
ax.legend()
plt.savefig(f'{PLOTS_DIRECTORY}real_vs_bs_price.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
master_df.groupby('Moneyness')['AbsolutePctError'].mean().sort_values().plot(kind='bar', ax=ax, color=['skyblue', 'salmon', 'lightgreen'])
ax.set_title('MAPE by Moneyness'); ax.set_xlabel('Moneyness')
ax.set_ylabel('Mean Absolute Percentage Error (%)')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIRECTORY}error_by_moneyness.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
master_df.groupby('ExpiryBin', observed=True)['AbsolutePctError'].mean().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('MAPE by Time to Expiry')
ax.set_xlabel('Time to Expiry')
ax.set_ylabel('Mean Absolute Percentage Error (%)')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIRECTORY}error_by_expiry.png')
plt.close()

near_term_expiry = master_df['ExpiryDate'].min()
smile_data = master_df[master_df['ExpiryDate'] == near_term_expiry]
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(smile_data[smile_data['OptionType'] == 'Call']['StrikePrice'], smile_data[smile_data['OptionType'] == 'Call']['IV'], 'o-', label='Calls IV')
ax.plot(smile_data[smile_data['OptionType'] == 'Put']['StrikePrice'], smile_data[smile_data['OptionType'] == 'Put']['IV'], 'o-', label='Puts IV')
ax.axvline(x=NIFTY_SPOT_PRICE, color='r', linestyle='--', label=f'Spot Price ({NIFTY_SPOT_PRICE})')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title(f'Implied Volatility Smile for {near_term_expiry.strftime("%d-%b-%Y")} Expiry')
ax.legend(); ax.grid(True)
plt.savefig(f'{PLOTS_DIRECTORY}iv_smile.png')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(smile_data['StrikePrice'], smile_data['Error'], width=20, alpha=0.7, label='Pricing Error (Real - BS)')
ax.axhline(y=0, color='r', linestyle='-')
ax.axvline(x=NIFTY_SPOT_PRICE, color='k', linestyle='--', label='Spot Price')
ax.set_xlabel('Strike Price'); ax.set_ylabel('Pricing Error (₹)')
ax.set_title(f'Pricing Error vs. Strike Price for {near_term_expiry.strftime("%d-%b-%Y")} Expiry'); ax.legend()
ax.grid(True)
plt.savefig(f'{PLOTS_DIRECTORY}error_vs_strike.png')
plt.close()

heatmap_data = master_df.pivot_table(values='AbsolutePctError', index='Moneyness', columns='ExpiryBin', aggfunc='mean', observed=True)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", ax=ax, linewidths=.5)
ax.set_title('Heatmap of Mean Percentage Error')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIRECTORY}error_heatmap.png')
plt.close()

output_filename = 'full_analysis_all_expiries.csv'
final_columns = ['ExpiryDate', 'Moneyness', 'ExpiryBin', 'OptionType', 'StrikePrice', 'UnderlyingPrice', 'RealPrice', 'IV', 'TimeToExpiry', 'BS_Price', 'Error', 'AbsolutePctError']
master_df[final_columns].to_csv(output_filename, index=False, float_format='%.2f')

print(f"\nAnalysis Complete! All results saved to '{output_filename}', plots saved to '{PLOTS_DIRECTORY}' and report saved to 'README.md'.")