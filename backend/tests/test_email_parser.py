from app.ingestion.parsers.email_parser import EmailParser


sample_email = """
Hi Support,

I cancelled my subscription two weeks ago.

My refund has still not arrived.

Please check the payment status.

Regards,

John Doe
"""


def main():

    parser = EmailParser()

    result = parser.parse(sample_email)

    print("=" * 70)
    print("Email Parser Test")
    print("=" * 70)

    for key, value in result.items():

        print(f"{key}:")
        print(value)
        print("-" * 40)

    print("=" * 70)


if __name__ == "__main__":
    main()