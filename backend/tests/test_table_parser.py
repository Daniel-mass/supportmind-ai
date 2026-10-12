from app.ingestion.parsers.table_parser import TableParser


sample_table = """
Plan,Price,Refund,Support

Basic,299,No,Email
Premium,799,Yes,24x7 Chat
Enterprise,Custom,Yes,Dedicated Manager
"""


def main():

    parser = TableParser()

    rows = parser.parse(sample_table)

    print("=" * 70)
    print("Table Parser Test")
    print("=" * 70)

    print(f"Rows Parsed : {len(rows)}")

    print("-" * 70)

    for row in rows:

        print(row)

    print("=" * 70)


if __name__ == "__main__":
    main()