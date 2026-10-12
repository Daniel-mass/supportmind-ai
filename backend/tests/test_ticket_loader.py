from pathlib import Path

from app.ingestion.loaders.ticket_loader import TicketLoader


def main():

    loader = TicketLoader()

    tickets = loader.load(
        Path(r"C:\Users\danie\supportmind-ai\data\raw\documents\sample_ticket.csv")
    )

    print("=" * 70)
    print("Ticket Loader Test")
    print("=" * 70)

    print(f"Total Tickets Loaded : {len(tickets)}")

    print("-" * 70)

    for ticket in tickets:

        print(ticket.metadata)
        print(ticket.content)
        print("-" * 70)


if __name__ == "__main__":
    main()