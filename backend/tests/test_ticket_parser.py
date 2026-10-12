from app.ingestion.parsers.ticket_parser import TicketParser


sample_ticket = {

    "TicketID": "T001",

    "Customer": "John Doe",

    "Category": "Refund",

    "Priority": "High",

    "Status": "Open",

    "Description": "Refund has not arrived."
}


def main():

    parser = TicketParser()

    result = parser.parse(sample_ticket)

    print("=" * 70)
    print("Ticket Parser Test")
    print("=" * 70)

    for key, value in result.items():

        print(f"{key:<15}: {value}")

    print("=" * 70)


if __name__ == "__main__":
    main()