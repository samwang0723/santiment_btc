import pandas as pd
import mplfinance as mpf
import numpy as np


def get_btc_data() -> pd.DataFrame:
    btc_price = pd.read_csv("data/market-price.csv", thousands=",", quotechar='"')
    # Convert percentage strings to float
    btc_price["Change %"] = btc_price["Change %"].str.rstrip("%").astype("float")
    # Convert volume strings to float using the custom function
    btc_price["Vol."] = btc_price["Vol."].apply(convert_volume)
    btc_price["Date"] = pd.to_datetime(btc_price["Date"])

    return btc_price


# Function to convert volume strings to float
def convert_volume(volume_str):
    if "K" in volume_str:
        return float(volume_str.rstrip("K")) * 1000
    elif "M" in volume_str:
        return float(volume_str.rstrip("M")) * 1000000
    elif "B" in volume_str:
        return float(volume_str.rstrip("B")) * 1000000000
    else:
        return float(volume_str)


def save_analysis_data():
    btc_price = get_btc_data()
    df = pd.read_csv("data/btc_features.csv")
    btc_features = df.loc[
        :,
        [
            "dt",
            # "exchange_inflow_usd",
            # "exchange_outflow_usd",
            "sentiment_balance",
            "unique_social_volume_1h",
            # "transaction_volume_5min",
            "miners_to_exchanges_flow",
            "whale_transaction_count_more_than_100k_usd_5min",
            "whale_transaction_count_more_than_1m_usd_5min",
        ],
    ]

    btc_price["Date"] = pd.to_datetime(btc_price["Date"])
    btc_features["dt"] = pd.to_datetime(btc_features["dt"])

    btc_price["5_day_ma"] = btc_price["Price"].rolling(window=5).mean()
    btc_price["10_day_ma"] = btc_price["Price"].rolling(window=10).mean()
    btc_price["20_day_ma"] = btc_price["Price"].rolling(window=20).mean()

    btc_price["5_day_mv"] = btc_price["Vol."].rolling(window=5).mean()
    btc_price["10_day_mv"] = btc_price["Vol."].rolling(window=10).mean()
    btc_price["20_day_mv"] = btc_price["Vol."].rolling(window=20).mean()

    merged_data = btc_features.merge(
        btc_price, left_on="dt", right_on="Date", how="inner"
    )

    merged_data.to_csv("data/analysis.csv", index=False)


def simulate_sell(analyzed_data) -> pd.DataFrame:
    btc_price = get_btc_data()
    btc_price.sort_values(by="Date", inplace=True)

    max_profit = 0.10
    max_loss = -0.15
    sell_dates = []
    sell_prices = []

    # Loop through the DataFrame using iterrows()
    for _, row in analyzed_data.iterrows():
        buy_date = row["dt"]
        buy_price = row["Price"]
        for _, row2 in btc_price.iterrows():
            sell_date = row2["Date"]
            sell_price = row2["Price"]
            if sell_date > buy_date:
                profit = (sell_price - buy_price) / buy_price
                if profit > max_profit:
                    sell_dates.append(sell_date)
                    sell_prices.append(sell_price)
                    break
                elif profit < max_loss:
                    sell_dates.append(sell_date)
                    sell_prices.append(sell_price)
                    break

    # Create a DataFrame with date and price headers
    df = pd.DataFrame({"Date": sell_dates, "Sell_Price": sell_prices})
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def show_candlestick_chart(analyzed_data, potential_sell_data):
    btc_price = get_btc_data()
    btc_price.sort_values(by="Date", inplace=True)

    # Convert Date columns to datetime objects
    analyzed_data["Date"] = pd.to_datetime(analyzed_data["Date"])
    potential_sell_data["Date"] = pd.to_datetime(potential_sell_data["Date"])

    # Merge the two DataFrames on the Date column
    merged_data = pd.merge(btc_price, analyzed_data, on="Date", how="left")
    merged_data = pd.merge(merged_data, potential_sell_data, on="Date", how="left")

    # Drop duplicate rows based on the "Date" column
    merged_data.drop_duplicates(subset="Date", inplace=True)

    # Set the Date column as the index
    merged_data.set_index("Date", inplace=True)
    merged_data.to_csv("data/merged.csv", index=False)

    # Set the Date column as the index
    ohlc_data = merged_data[["Open_x", "High_x", "Low_x", "Price_x", "Vol._x"]].copy()
    ohlc_data.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlc_data["Volume"] = ohlc_data["Volume"].astype(int)

    # Add markers for indicators
    indicator_dates = merged_data.loc[~merged_data["dt"].isnull()].index
    indicator_prices = merged_data.loc[~merged_data["dt"].isnull(), "Price_x"]

    # Create a DataFrame with the marker data
    marker_data = pd.DataFrame(index=ohlc_data.index)
    marker_data["Marker"] = np.nan
    marker_data.loc[indicator_dates, "Marker"] = indicator_prices

    # Add custom markers to the plot
    mc = mpf.make_marketcolors(up="g", down="r", inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    apdict = mpf.make_addplot(
        marker_data, type="scatter", markersize=50, marker="o", color="blue"
    )

    # Add markers for indicators
    sindicator_dates = merged_data.loc[~merged_data["Sell_Price"].isnull()].index
    sindicator_prices = merged_data.loc[
        ~merged_data["Sell_Price"].isnull(), "Sell_Price"
    ]

    # Create a DataFrame with the marker data
    smarker_data = pd.DataFrame(index=ohlc_data.index)
    smarker_data["Marker_s"] = np.nan
    smarker_data.loc[sindicator_dates, "Marker_s"] = sindicator_prices

    # Add custom markers to the plot
    sapdict = mpf.make_addplot(
        smarker_data, type="scatter", markersize=50, marker="o", color="orange"
    )

    mpf.plot(
        ohlc_data,
        type="candle",
        style=s,
        figsize=(12, 8),
        addplot=[apdict, sapdict],
    )


def filter_analysis_data() -> pd.DataFrame:
    # Read the CSV data
    analysis = pd.read_csv("data/analysis.csv")

    # Create a condition for the moving averages trend going up
    ma_trend_up = (
        (analysis["5_day_ma"].shift(-3) > analysis["5_day_ma"].shift(-2))
        & (analysis["10_day_ma"].shift(-3) > analysis["10_day_ma"].shift(-2))
        & (analysis["20_day_ma"].shift(-3) > analysis["20_day_ma"].shift(-2))
        & (analysis["5_day_ma"].shift(-2) > analysis["5_day_ma"].shift(-1))
        & (analysis["10_day_ma"].shift(-2) > analysis["10_day_ma"].shift(-1))
        & (analysis["20_day_ma"].shift(-2) > analysis["20_day_ma"].shift(-1))
        & (analysis["5_day_ma"].shift(-1) > analysis["5_day_ma"])
        & (analysis["10_day_ma"].shift(-1) > analysis["10_day_ma"])
        & (analysis["20_day_ma"].shift(-1) > analysis["20_day_ma"])
        & (analysis["Vol."] > analysis["5_day_mv"])
        & (analysis["Vol."] > analysis["10_day_mv"])
        & (analysis["Vol."] > analysis["20_day_mv"])
    )

    # Create a condition for sentiment_balance, whale_transaction_count_more_than_100k_usd_5min, and whale_transaction_count_more_than_1m_usd_5min
    other_conditions = (
        analysis["sentiment_balance"]
        >= 20 & (analysis["whale_transaction_count_more_than_100k_usd_5min"] >= 250)
        # & (analysis["whale_transaction_count_more_than_1m_usd_5min"] >= 100)
    )

    # Combine the conditions and filter the data
    filtered_data = analysis[ma_trend_up & other_conditions]
    filtered_data["dt"] = pd.to_datetime(filtered_data["dt"])

    return filtered_data


# Calculation
save_analysis_data()
filtered_data = filter_analysis_data()
potential_sell_data = simulate_sell(filtered_data)
show_candlestick_chart(filtered_data, potential_sell_data)
