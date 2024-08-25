import os
import json
import pandas as pd

class PriceProcessor:
    def __init__(self, input_dir=None, interim_dir=None, output_dir=None):
        
        self.raw_data_dir = input_dir
        self.interim_data_dir = interim_dir
        self.processed_data_dir = output_dir

        self.trading_dates_path = os.path.join(self.interim_data_dir, "trading_dates.parquet")
        self.wasde_data_path = os.path.join(self.interim_data_dir,"wasde_soybeans.parquet")
        
        self.aggregate_price_path = os.path.join(self.interim_data_dir, "prices_soybeans_aggregate.parquet")
        self.continuous_price_path = os.path.join(self.interim_data_dir, "prices_soybeans_continuous.parquet")
        self.continuous_price_ext_path = os.path.join(self.interim_data_dir, "prices_soybeans_continuous_ext.parquet")
        
        self.model_training_data_path = os.path.join(self.processed_data_dir, "soybeans_model_training_data.parquet")
        self.model_training_csv_path = os.path.join(self.processed_data_dir, "soybeans_model_training_data.csv")
        self.__contract_sequence = ["SF", "SH", "SK", "SN", "SQ", "SU", "SX"]

    @staticmethod
    def unpack_prices(price_json):
        if not price_json or price_json == '{}':
            return None, None
        
        try:
            price_dict = json.loads(price_json)
            high = price_dict.get("high", None)
            low = price_dict.get("low", None)
            return float(high), float(low) if high is not None and low is not None else (None, None)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None, None

    @staticmethod
    def determine_contract(month, year, date, wasde_date):
        yr = str(year)[-2:]  # Get the last two digits of the year
        contracts = {
            1: (f"SF{yr}", f"SH{yr}"),
            2: (f"SH{yr}", f"SH{yr}"),
            3: (f"SH{yr}", f"SK{yr}"),
            4: (f"SK{yr}", f"SK{yr}"),
            5: (f"SK{yr}", f"SN{yr}"),
            6: (f"SN{yr}", f"SN{yr}"),
            7: (f"SN{yr}", f"SQ{yr}"),
            8: (f"SQ{yr}", f"SU{yr}"),
            9: (f"SU{yr}", f"SX{yr}"),
            10: (f"SX{yr}", f"SX{yr}"),
            11: (f"SX{yr}", f"SF{str(int(year) + 1)[-2:]}"),
            12: (f"SF{str(int(year) + 1)[-2:]}", f"SF{str(int(year) + 1)[-2:]}"),
        }
        
        current_contract, next_contract = contracts.get(month, ("", ""))
        
        if date >= wasde_date:
            return next_contract
        else:
            return current_contract

    @staticmethod
    def generate_price_data(row):
        price_high = row["High"]
        price_low = row["Low"]
        
        if pd.isna(price_high) or pd.isna(price_low):
            return json.dumps({})
        else:
            return json.dumps({"high": float(price_high), "low": float(price_low)})
    
    def get_next_contracts(self, base_contract):
        year_suffix = base_contract[-2:]
        base_contract_name = base_contract[:-2]
        base_index = self.__contract_sequence.index(base_contract_name)

        contracts = [f"{contract}{year_suffix}" for contract in self.__contract_sequence[base_index:]]
        if len(contracts) < 7:
            next_year_suffix = f"{(int(year_suffix) + 1) % 100:02}"
            contracts.extend(
                [f"{contract}{next_year_suffix}" for contract in self.__contract_sequence[: 7 - len(contracts)]]
            )
        return contracts[:7]
    
    def get_sorted_contract_names(self, csv_files, years):
        contracts_sorted_by_expiration = []
        for year in years:
            for symbol in self.__contract_sequence:
                contract_name = f"{symbol}{str(year)[-2:]}.csv"
                if contract_name in csv_files:
                    contracts_sorted_by_expiration.append(contract_name)
        return contracts_sorted_by_expiration

    def process_raw_price_data(self, contracts_sorted_by_expiration, all_price_data):
        all_price_data['Date'] = pd.to_datetime(all_price_data['Date'])
        for contract in contracts_sorted_by_expiration:
            contract_name = contract.replace(".csv", "")
            path = os.path.join(self.raw_data_dir, contract)

            contract_price_data = pd.read_csv(path).iloc[:-1]
            contract_price_data = contract_price_data.rename(columns={"Time": "Date"})
            contract_price_data["Date"] = pd.to_datetime(contract_price_data["Date"])
            contract_price_data["Price"] = contract_price_data.apply(self.generate_price_data, axis=1)

            all_price_data = all_price_data.merge(
                contract_price_data[["Date", "Price"]],
                on="Date",
                how="left",
                suffixes=("", f"_{contract_name}"),
            )
            all_price_data.rename(columns={"Price": f"{contract_name}"}, inplace=True)

        return all_price_data.fillna("")

    def process_continuous_data(self, trading_dates, wasde_dates, price_data):
        continuous_data = []
        wasde_iter = iter(wasde_dates["Report Date"])
        current_wasde_date = next(wasde_iter, None)
        contract_name = None

        for date in trading_dates["Date"]:
            if current_wasde_date is not None and date >= current_wasde_date:
                month, year = date.month, date.year
                contract_name = self.determine_contract(month, year, date, current_wasde_date)
                
                try:
                    current_wasde_date = next(wasde_iter)
                except StopIteration:
                    current_wasde_date = wasde_dates["Report Date"].iloc[-1]
            
            if contract_name and not price_data.loc[price_data["Date"] == date, contract_name].empty:
                daily_price_json = price_data.loc[price_data["Date"] == date, contract_name].values[0]
                high, low = self.unpack_prices(daily_price_json)

                if high is not None and low is not None:
                    average = (high + low) / 2
                    continuous_data.append([date, contract_name, high, low, average])

        return pd.DataFrame(
            continuous_data, columns=["Date", "Contract", "High", "Low", "Average"]
        )

    def process_year_ahead_pricing_data(self, trading_dates, wasde_dates, price_data):
        continuous_data_ext = []
        wasde_iter = iter(wasde_dates["Report Date"])
        current_wasde_date = next(wasde_iter, None)
        contract_name = None
        contract_collection = []

        for date in trading_dates["Date"]:
            if current_wasde_date is not None and date >= current_wasde_date:
                month, year = date.month, date.year
                contract_name = self.determine_contract(month, year, date, current_wasde_date)
                contract_collection = self.get_next_contracts(contract_name)

                try:
                    current_wasde_date = next(wasde_iter)
                except StopIteration:
                    current_wasde_date = wasde_dates["Report Date"].iloc[-1]

            for contract in contract_collection:
                if contract and not price_data.loc[price_data["Date"] == date, contract].empty:
                    daily_price_json = price_data.loc[price_data["Date"] == date, contract].values[0]
                    high, low = self.unpack_prices(daily_price_json)

                    if high is not None and low is not None:
                        average = (high + low) / 2
                        continuous_data_ext.append([date, contract, high, low, average])
                        
        return pd.DataFrame(
            continuous_data_ext, columns=["Date", "Contract", "High", "Low", "Average"]
        )
    
    def aggregate_price_data(self):
        trading_dates = pd.read_parquet(self.trading_dates_path)

        data = pd.DataFrame(trading_dates, columns=["Date"])
        symbols = ["SF", "SH", "SK", "SN", "SQ", "SU", "SX"]
        years = list(range(2000, pd.Timestamp.today().year + 2))

        contract_csv_collection = [f for f in os.listdir(self.raw_data_dir) if f.endswith(".csv")]
        contracts_sorted = self.get_sorted_contract_names(contract_csv_collection, symbols, years)

        data = self.process_raw_price_data(self.raw_data_dir, contracts_sorted, data)
        data.to_parquet(self.aggregate_price_path, index=False)

    def generate_continuous_price_data(self):
        price_data = pd.read_parquet(self.aggregate_price_path)
        trading_dates = pd.read_parquet(self.trading_dates_path)
        wasde_dates = pd.read_parquet(self.wasde_data_path)

        trading_dates["Date"] = pd.to_datetime(trading_dates["Date"])
        wasde_dates["Report Date"] = pd.to_datetime(wasde_dates["Report Date"])

        price_continuous = self.process_continuous_data(trading_dates, wasde_dates, price_data)
        price_continuous.to_parquet(self.continuous_price_path, index=False)
        
        price_ext = self.process_year_ahead_pricing_data(trading_dates, wasde_dates, price_data)
        price_ext.to_parquet(self.continuous_price_ext_path, index=False)

    def aggregate_model_input_data(self):
        processed_data = pd.read_parquet(self.model_training_data_path)
        daily_price_data = pd.read_parquet(self.continuous_price_path)

        # Standardize date columns to datetime.date
        processed_data.rename(columns={"Report Date": "Date"}, inplace=True)
        processed_data["Date"] = pd.to_datetime(processed_data["Date"]).dt.date
        daily_price_data["Date"] = pd.to_datetime(daily_price_data["Date"]).dt.date

        # Initialize lists to store results
        price_high_list = []
        price_low_list = []
        price_average_list = []
        price_collections = []

        # Create an iterator over the report dates
        report_date_iter = iter(processed_data["Date"])

        # Get the first report date
        start_date = next(report_date_iter, None)

        # Iterate through report dates
        for end_date in report_date_iter:
            # Filter the daily_price_data for the date range
            group = daily_price_data[(daily_price_data["Date"] >= start_date) & (daily_price_data["Date"] < end_date)]

            if not group.empty:
                group = group.head(15)  # Limit to the first 15 rows
                price_array = group["Average"].tolist()

                price_high = group["High"].max()
                price_low = group["Low"].min()
                price_average = group["Average"].mean()
            else:
                # If the group is empty, assign NaN or an appropriate default value
                price_array = []
                price_high = float("nan")
                price_low = float("nan")
                price_average = float("nan")

            # Append the results to the respective lists
            price_high_list.append(float(price_high))
            price_low_list.append(float(price_low))
            price_average_list.append(float(price_average))
            price_collections.append(price_array)

            # Update the start date to the current end date for the next iteration
            start_date = end_date

        # Handle the last report date (no next end date)
        group = daily_price_data[daily_price_data["Date"] >= start_date]

        if not group.empty:
            group = group.head(15)
            price_array = group["Average"].tolist()

            price_high = group["High"].max()
            price_low = group["Low"].min()
            price_average = group["Average"].mean()
        else:
            price_array = []
            price_high = float("nan")
            price_low = float("nan")
            price_average = float("nan")

        # Append the results to the lists for the last report date
        price_high_list.append(float(price_high))
        price_low_list.append(float(price_low))
        price_average_list.append(float(price_average))
        price_collections.append(price_array)

        # Add the new columns to the DataFrame
        processed_data["Price_High"] = price_high_list
        processed_data["Price_Low"] = price_low_list
        processed_data["Price_Average"] = price_average_list
        processed_data["Average_Price_Collection"] = price_collections

        # Save the DataFrame to CSV
        processed_data.to_parquet(self.model_training_data_path)
        processed_data.to_csv(self.model_training_csv_path, index=False)