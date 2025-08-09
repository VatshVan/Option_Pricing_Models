
# Black-Scholes Model Accuracy Report

### Analysis Parameters
- **Analysis Date:** 2025-08-08
- **Nifty Spot Price:** ₹24,363.30
- **Risk-Free Rate:** 6.90%

---

### Overall Model Error
This value shows the average percentage difference between the model's price and the real market price.

- **Mean Absolute Percentage Error (MAPE): 8.33%**

---

## Detailed Error Breakdown

### MAPE by Option Type
This table shows the model's average error for Call options vs. Put options.

| Option Type               | MAPE (%)   |
|---------------------------|------------|
| Call                      | 8.56       |
| Put                       | 8.01       |

### MAPE by Moneyness
This table shows the model's average error based on whether an option is In-the-Money (ITM), At-the-Money (ATM), or Out-of-the-Money (OTM).

| Moneyness                 | MAPE (%)   |
|---------------------------|------------|
| In-the-Money (ITM)        | 3.29       |
| At-the-Money (ATM)        | 7.41       |
| Out-of-the-Money (OTM)    | 10.26      |

### MAPE by Time to Expiry
This table shows the model's average error for options with different expiration timeframes.

| Expiry Bin                | MAPE (%)   |
|---------------------------|------------|
| Near-Term (0-30 days)     | 4.63       |
| Mid-Term (31-90 days)     | 14.85      |
| Long-Term (>90 days)      | 24.70      |

