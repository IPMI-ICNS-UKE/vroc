from pathlib import Path

from vroc.database.client import DatabaseClient

folder = Path("/datalake/learn2reg/runs")

output_filepath = Path("/datalake/learn2reg/merged_runs.sqlite")

database_filepaths = sorted(folder.glob("*.sqlite"))


client = DatabaseClient(output_filepath)

for database_filepath in database_filepaths:
    print(f"Attach database {database_filepath.name}")
    client.database.attach(filename=database_filepath, name="other")

    for table in client.database.get_tables():
        print(f"Updating table {table}")
        client.database.execute_sql(
            f"INSERT OR IGNORE INTO {table} SELECT * FROM other.{table};"
        )

    client.database.detach("other")

client.database.close()
