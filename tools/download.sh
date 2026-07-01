#!/usr/bin/env bash

function kaggle_download() {
  if [ -z "$1" ]; then
    echo "Usage: $0 <kaggle_project_name>"
    exit 1
  fi
  local kaggle_project_name=$1

  local root="$(git rev-parse --show-toplevel)"
  local directories=("data" "models" "submissions" "src")
  mkdir -p "$root/$kaggle_project_name"
  for dir in "${directories[@]}"; do
    mkdir -p "$root/$kaggle_project_name/$dir"
  done

  uv run kaggle competitions download -c "$kaggle_project_name" -p "$root/$kaggle_project_name/data"
  local zip_files=`find "$root/$kaggle_project_name/data" -name "*.zip"`
  for zip_file in $zip_files; do
    unzip "$zip_file" -d "$root/$kaggle_project_name/data"
    rm "$zip_file"
  done
  (
    cd "$root/$kaggle_project_name/"
    uv init
  )
}

function put_files_into_duckdb() {
    local root="$(git rev-parse --show-toplevel)"
    local data_dir="$root/$1/data"
    local db_path="$root/$1/data.duckbb"
    
    if [ ! -d "$data_dir" ]; then
        echo "Data directory not found: $data_dir"
        exit 1
    fi
    csv_files=$(find "$data_dir" -type f -name "*.csv")
    if [ -z "$csv_files" ]; then
        echo "No CSV files found in $data_dir"
        exit 1
    fi
    for csv_file in $csv_files; do
        table_name=$(basename "$csv_file" .csv)
        echo "Importing $csv_file into DuckDB as table $table_name"
        duckdb "$db_path" -c "CREATE TABLE IF NOT EXISTS $table_name AS SELECT * FROM read_csv_auto('$csv_file');"
    done
}

kaggle_download "$1"
put_files_into_duckdb "$1"
