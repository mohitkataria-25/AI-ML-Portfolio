from __future__ import  annotations

import argparse
import pandas as pd
import logging
import yaml
import io
import boto3

from pathlib import Path
from typing import Tuple, Mapping, Optional
from sklearn.model_selection import train_test_split

def split_feature_target(df, target_col):

    x = df.drop(target_col, axis=1)
    y = df[target_col]

    return x, y

def clean_data (df: pd.DataFrame):
    
    null_cols = null_column_check(df=df)
    
    if len(null_cols) != 0:
        for column in null_cols:
            df = df[df[column].notna()]
    return df
    

def null_column_check(df: pd.DataFrame):
    
    null_cols = []
    for col, count in df.isnull().sum().items():
        if count != 0: 
            null_cols.append(col)
    return null_cols

def load_data(config):

    
    try:
        raw_config = config["data_ingest"]
    except:
        raise ValueError("Raw file path not configured")
    raw_path_value = raw_config.get("raw_path")
    
    if not raw_path_value:
        raise ValueError('config[data_ingest][raw_path] not set')
    
    raw_data = Path(raw_path_value)
    
    if not raw_data.exists():
        raise FileNotFoundError (f'Raw data file not found at {raw_data.resolve()}')
    logging.info('Loading raw data from %s', raw_data) 
    
    data = pd.read_csv(raw_data)

    logging.info('data loaded from %s', raw_data)

    data = clean_data(data)

    return data
 
def train_val_test_split(x, y, config):

    #extract trainng dateset
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, random_state=42, stratify=y, train_size=config["data_ingest"]["train_size"])

    #extract validation and test dataset
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, random_state=42, stratify=y_val_test, test_size=config["data_ingest"]["test_size"])
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def save_splits(x_train, x_val, x_test, y_train, y_val, y_test, config):
    
    output_cfg = config["data_ingest"].get("output", {})
    file_format = output_cfg.get("format", "parquet").lower()
    storage_type = output_cfg.get("storage_type", "local")

    if storage_type == "local":
        _save_splits_local(x_train, x_val, x_test, y_train, y_val, y_test, output_cfg, file_format)
    elif storage_type == "s3":
        _save_splits_s3(x_train, x_val, x_test, y_train, y_val, y_test, output_cfg, file_format)
    else:
        raise ValueError(f"Unknown storage_type: {storage_type}")
        
def _save_splits_local(x_train, x_val, x_test, y_train, y_val, y_test, output_cfg, file_format: str):
    
    base_dir = Path(output_cfg.get("local_dir", "data/ingested"))
    base_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving splits locally to %s {base_dir}")

    suffix = ".parquet" if file_format == "parquet" else ".csv"

    if file_format == "parquet":
        x_train.to_parquet(base_dir / f"x_train{suffix}", index=False)
        x_val.to_parquet(base_dir / f"x_val{suffix}", index=False)
        x_test.to_parquet(base_dir / f"x_test{suffix}", index=False)

        y_train.to_frame("Target").to_parquet(base_dir / f"y_train{suffix}", index=False)
        y_val.to_frame("Target").to_parquet(base_dir / f"y_val{suffix}", index=False)
        y_test.to_frame("Target").to_parquet(base_dir / f"y_test{suffix}", index=False)
    else:
        x_train.to_csv(base_dir / f"x_train{suffix}", index=False)
        x_val.to_csv(base_dir / f"x_val{suffix}", index=False)
        x_test.to_csv(base_dir / f"x_test{suffix}", index=False)

        y_train.to_frame("Target").to_csv(base_dir / f"y_train{suffix}", index=False)
        y_val.to_frame("Target").to_csv(base_dir / f"y_val{suffix}", index=False)
        y_test.to_frame("Target").to_csv(base_dir / f"y_test{suffix}", index=False)

    logging.info("Local splits saved")

def _save_splits_s3(x_train, x_val, x_test, y_train, y_val, y_test, output_cfg, file_format: str):
    bucket = output_cfg.get("s3_bucket")
    prefix = output_cfg.get("s3_prefix", "datasets/renewind/splits")
    
    if not bucket:
            raise ValueError("config['data_ingest']['output']['s3_bucket'] must be set for S3 storage")
    s3 = boto3.client('s3')

    if file_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported file format for S3: {file_format}")

    logging.info("Saving splits to s3://%s/%s", bucket, prefix)

    def df_to_s3(df, key):
        if file_format == "csv":
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        else:
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            s3.put_object(Bucket=bucket, Key=key, Body=parquet_buffer.getvalue())

    ext = ".csv" if file_format == "csv" else ".parquet"

    df_to_s3(x_train, f"{prefix}/x_train{ext}")
    df_to_s3(x_val, f"{prefix}/x_val{ext}")
    df_to_s3(x_test, f"{prefix}/x_test{ext}")

    df_to_s3(y_train, f"{prefix}/y_train{ext}")
    df_to_s3(y_val, f"{prefix}/y_val{ext}")
    df_to_s3(y_test, f"{prefix}/y_test{ext}")
    
    logging.info("S3 splits saved")


def main(config_path):
    
    print(f"[data_ingest] Using config {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[data_ingest] loading raw data....")
    df = load_data(config=config)

    print (f"[data_ingest] Perform data clean up....")
    df_cleaned = clean_data(df=df)

    print(f"[data_ingest] Extracting Predictors and target features....")
    pred_features, tar_features = split_feature_target(df=df_cleaned, target_col="Target")

    print(f"[data_ingest] Spliting the dataset into train, validation and test datesets....")
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x=pred_features, y=tar_features, config=config)

    print (f"[data_ingest] Saving extracted datasets....")
    save_splits(x_train, x_val, x_test, y_train, y_val, y_test, config)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Renewind = Data-pipeline")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to the configurations for the current run"
    )
    args = parser.parse_args()
    main(config_path=args.config)