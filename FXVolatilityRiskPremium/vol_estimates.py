import numpy as np

def parkinson(price_data, window=30, day_count=252):
    log_hl = np.log(price_data['High']/price_data['Low'])
    sum_log_hl_sq = log_hl.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True)    
    volatility = np.sqrt(day_count) * np.sqrt((1 / (4 * np.log(2))) * (1 / window) * sum_log_hl_sq)
    return volatility

def rodgers_satchell(price_data, window=30, day_count=252):
    log_hc = np.log(price_data['High'] / price_data['Close'])
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lc = np.log(price_data['Low'] / price_data['Close'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    sum_prod = (log_hc * log_ho + log_lc * log_lo).rolling(window).sum()
    volatility = np.sqrt(day_count) * np.sqrt((1 / (window)) * sum_prod)
    return volatility

def garman_klass(price_data, window=30, day_count=252):
    log_hl = np.log(price_data['High']/price_data['Low'])
    log_ctc = np.log(price_data['Close']/price_data['Close'].shift(1))
    sum_log_hl_sq = log_hl.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True)
    sum_log_ctc_sq = log_ctc.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True)
    volatility = np.sqrt(day_count) * np.sqrt((1 / (window)) * (0.5 * sum_log_hl_sq - (2 * np.log(2) - 1) *
                                                            sum_log_ctc_sq))
    return volatility

def yang_zhang(price_data, window=30, day_count=252):
    log_olc = np.log(price_data['Open']/price_data['Close'].shift(1))
    log_clo = np.log(price_data['Close']/price_data['Open'].shift(1))
    sigma_o_sq = (1 / (window - 1)) * log_olc.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True)
    sigma_c_sq = (1 / (window - 1)) * log_clo.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True)
    sigma_rs_sq = (window / (day_count * (window - 1))) * \
    rodgers_satchell(price_data, window=window, day_count=day_count) ** 2
    k = 0.34 / (1 + (window + 1) / (window - 1))
    volatility = np.sqrt(day_count) * np.sqrt(sigma_o_sq  + k * sigma_c_sq + (1 - k) * sigma_rs_sq)
    return volatility