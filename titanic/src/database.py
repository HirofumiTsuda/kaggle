import pathlib
import sqlite3
import pandas as pd

SQLITE3_FILE = pathlib.Path(__file__).parents[1] / "data" / "titanic.db"


def write_to_sqlite3(data: dict[str, pd.DataFrame]) -> None:
    with sqlite3.connect(SQLITE3_FILE) as conn:
        for table_name, df in data.items():
            try:
                df.to_sql(table_name, conn, if_exists="fail", index=False)
            except ValueError as e:
                print(e)
