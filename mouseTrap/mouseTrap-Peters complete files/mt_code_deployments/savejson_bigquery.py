import argparse

from google.cloud import bigquery

def load_data_from_file(dataset_name, table_name, source_file_name):
    bigquery_client = bigquery.Client()
    dataset = bigquery_client.dataset(dataset_name)
    table = dataset.table(table_name)

    # Reload the table to get the schema.
    table.reload()

    with open(source_file_name, 'rb') as source_file:
        # This example uses CSV, but you can use other formats.
        # See https://cloud.google.com/bigquery/loading-data
        job = table.upload_from_file(
            source_file, source_format='text/csv')

    job.result()  # Wait for job to complete

    print('Loaded {} rows into {}:{}.'.format(
        job.output_rows, dataset_name, table_name))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_name')
    parser.add_argument('table_name')
    parser.add_argument(
        'source_file_name', help='Path to a .csv file to upload.')

    args = parser.parse_args()

    load_data_from_file(
        args.dataset_name,
        args.table_name,
        args.source_file_name)

main()