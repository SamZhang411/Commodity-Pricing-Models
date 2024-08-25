import pandas as pd
from pathlib import Path

class WASDEProcessor:
    def __init__(self, excel_path=None, csv_paths=None, csv_dir=None):
        self.wasde_0010_path = excel_path
        self.wasde_1020_path = csv_paths
        self.wasde_2124_path = csv_dir

    def filter_by_commodity(self, data):
        filter_conditions = [
            ("World Corn Supply and Use", ["World", "Major exporters", "United States", "Major Exporters"]),
            ("World Wheat Supply and Use", ["World", "Major exporters", "United States", "Major Exporters"]),
            ("World Soybean Supply and Use", ["World", "Argentina", "Brazil", "United States", "Major Exporters"]),
            ("World Soybean Meal Supply and Use", ["World", "Major exporters", "United States", "Major Exporters"]),
            ("World Soybean Oil Supply and Use", ["World", "Major exporters", "United States", "Major Exporters"]),
        ]

        filtered_data = []
        for title, regions in filter_conditions:
            filtered = data[(data["ReportTitle"] == title) & (data["Region"].isin(regions))]
            filtered_data.append(filtered)
        return filtered_data

    def process_csv_by_path(self, input_path, output_data):
        data = pd.read_csv(input_path, low_memory=False)
        data_collection = self.filter_by_commodity(data)

        for data_set in data_collection:
            data_set = data_set[data_set["ProjEstFlag"] == "Proj."]
            grouped_data = data_set.groupby("ReleaseDate")

            for _, group in grouped_data:
                filtered_data = group[["ReleaseDate", "Commodity", "Region", "Attribute", "Value", "Unit"]]
                filtered_data = filtered_data.rename(columns={"ReleaseDate": "Report Date"})
                output_data = pd.concat([output_data, filtered_data], ignore_index=True)

        return output_data

    def clean_data(self, data):
        conversion_dict = {"Domestic Feed": "Feed", "Domestic Crush": "Crush", "Domestic Total": "Total Use"}

        data["Report Date"] = pd.to_datetime(data["Report Date"]).dt.date
        data["Attribute"] = data["Attribute"].replace(conversion_dict)

        data_collection = data.groupby(["Report Date", "Commodity"])
        sorted_data = pd.concat([group for _, group in data_collection]).sort_values(by=["Report Date", "Commodity"])
        
        dtype_wasde_data = {
            "Report Date": "datetime64[ns]",
            "Commodity": "category",
            "Region": "category",
            "Attribute": "category",
            "Value": "float64"
        }
        sorted_data = sorted_data.astype(dtype_wasde_data)

        return sorted_data.reset_index(drop=True)

    def process_wasde_data(self, output_path):
        wasde_raw = pd.read_excel(self.wasde_0010_path, sheet_name=None)
        wasde_data = pd.concat(wasde_raw.values(), ignore_index=True)
        wasde_data.rename(columns={"Country": "Region"}, inplace=True)

        processed_data = pd.DataFrame()
        for path in self.wasde_1020_path:
            processed_data = self.process_wasde_by_path(path, processed_data)

        for file_path in Path(self.wasde_2124_path).glob("*.csv"):
            processed_data = self.process_wasde_by_path(file_path, processed_data)

        wasde_data = pd.concat([wasde_data, processed_data], ignore_index=True)
        wasde_data = self.clean_data(wasde_data)
        wasde_data.to_parquet(output_path, index=False)

    def aggregate_wasde_data(self, input_path, output_path):
        processed_data = pd.read_parquet(input_path)
        processed_data = processed_data.drop(columns=["Unit"])
        processed_rows = []

        replacements = {
            "OilSeed, Soybeans": "Soybeans",
            "Oilseed, Soybean": "Soybeans",
            "Meal, Soybeans": "Soybean Meal",
            "Oil, Soybeans": "Soybean Oil",
        }

        grouped = processed_data.groupby(["Report Date", "Commodity", "Region"])

        for (report_date, commodity, region), group in grouped:
            aggregate_row = {
                "Report Date": report_date,
                "Commodity": commodity,
                "Region": region,
                "Beginning Stocks": None,
                "Production": None,
                "Imports": None,
                "Exports": None,
                "Feed/Crush": None,
                "Total Use": None,
                "Ending Stocks": None,
                "STU": None,
            }

            ending_stocks = None
            total_use = None
            stu = None
            for _, record in group.iterrows():
                attribute = record["Attribute"]
                value = record["Value"]

                if attribute == "Beginning Stocks":
                    aggregate_row["Beginning Stocks"] = value
                elif attribute == "Production":
                    aggregate_row["Production"] = value
                elif attribute == "Imports":
                    aggregate_row["Imports"] = value
                elif attribute == "Exports":
                    aggregate_row["Exports"] = value
                elif attribute in ["Feed", "Crush"]:
                    aggregate_row["Feed/Crush"] = value
                elif attribute in ["Total Use", "Use, Total"]:
                    aggregate_row["Total Use"] = value
                    total_use = value
                elif attribute == "Ending Stocks":
                    aggregate_row["Ending Stocks"] = value
                    ending_stocks = value

            if ending_stocks and total_use is not None:
                stu = round(ending_stocks / total_use, 4)
                aggregate_row["STU"] = stu

            processed_rows.append(aggregate_row)

        processed_data = pd.DataFrame(processed_rows)
        processed_data["Commodity"] = processed_data["Commodity"].replace(replacements)

        dtype_aggregate_data = {
            "Report Date": "datetime64[ns]",
            "Commodity": "category",
            "Region": "category",
            "Beginning Stocks": "float64",
            "Production": "float64",
            "Imports": "float64",
            "Exports": "float64",
            "Feed/Crush": "float64",
            "Total Use": "float64",
            "Ending Stocks": "float64",
            "STU": "float64"
        }

        processed_data = processed_data.astype(dtype_aggregate_data)
        processed_data.to_parquet(output_path, index=False)

    def filter_soybeans_wasde_data(self, data_path, output_path):
        data = pd.read_parquet(data_path)
        data["Report Date"] = pd.to_datetime(data["Report Date"])
        data["Report Month"] = data["Report Date"].dt.to_period("M")
        filtered_rows = []

        grouped = data.groupby("Report Date")

        for report_date, group in grouped:
            soybean_row = {
                "Report Date": report_date,
                "Report Month": report_date.to_period("M"),
                "STU, US": None,
                "STU, AR": None,
                "STU, BR": None,
                "STU, Corn": None,
                "Production, US": None,
                "Production, AR": None,
                "Production, BR": None,
            }

            for _, row in group.iterrows():
                if row["Commodity"] == "Soybeans":
                    if row["Region"] == "United States":
                        soybean_row["STU, US"] = row["STU"]
                        soybean_row["Production, US"] = row["Production"]
                    elif row["Region"] == "Argentina":
                        soybean_row["STU, AR"] = row["STU"]
                        soybean_row["Production, AR"] = row["Production"]
                    elif row["Region"] == "Brazil":
                        soybean_row["STU, BR"] = row["STU"]
                        soybean_row["Production, BR"] = row["Production"]

                if row["Commodity"] == "Corn" and row["Region"] == "United States":
                    soybean_row["STU, Corn"] = row["STU"]

            filtered_rows.append(soybean_row)

        dtype_soybean_row = {
            "Report Date": "datetime64[ns]",
            "Report Month": "datetime64[ns]",
            "STU, US": "float64",
            "STU, AR": "float64",
            "STU, BR": "float64",
            "STU, Corn": "float64",
            "Production, US": "float64",
            "Production, AR": "float64",
            "Production, BR": "float64",
        }

        processed_data = pd.DataFrame(filtered_rows)
        processed_data = processed_data.astype(dtype_soybean_row)
        processed_data.to_parquet(output_path, index=False)

    def append_indicators(self, data_path, indicator_path, output_path):
        data = pd.read_parquet(data_path)
        indicators = pd.read_csv(indicator_path, low_memory=False)

        data["Report Month"] = pd.to_datetime(data["Report Date"]).dt.to_period("M")
        indicators["Date"] = pd.to_datetime(indicators["Date"]).dt.to_period("M")

        merged_data = pd.merge(
        data, indicators, how="left", left_on="Report Month", right_on="Date"
        )
    
        dtype_merged_data = {
            "Report Date": "datetime64[D]",
            "STU, US": "float64",
            "STU, AR": "float64",
            "STU, BR": "float64",
            "STU, Corn": "float64",
            "Production, US": "float64",
            "Production, AR": "float64",
            "Production, BR": "float64",
            "GDP (Bn USD)": "float64",
            "Gold": "float64",
            "DX": "float64",
            "Crude": "float64",
            "USD?BRL": "float64"
        }

        merged_data.drop(columns=["Report Month", "Date"], inplace=True)
        merged_data = merged_data.astype(dtype_merged_data)
        merged_data.to_parquet(output_path, index=False)