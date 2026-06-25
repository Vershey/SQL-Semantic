from __future__ import annotations

import sqlite3

from infrastructure.vector_store.in_memory_store import InMemoryVectorStore

_ORDERS = [
    ("Acme Corp", 1200.00, "2023-02-11"),
    ("Globex", 450.50, "2023-05-03"),
    ("Initech", 2999.99, "2023-09-21"),
    ("Umbrella", 175.00, "2023-11-30"),
    ("Acme Corp", 880.00, "2024-01-15"),
    ("Globex", 1320.75, "2024-02-02"),
    ("Stark Industries", 5400.00, "2024-03-19"),
    ("Initech", 60.00, "2024-04-08"),
    ("Wayne Enterprises", 2100.00, "2024-06-25"),
    ("Globex", 740.25, "2024-08-14"),
    ("Acme Corp", 310.00, "2024-09-30"),
    ("Stark Industries", 1990.00, "2024-12-05"),
]

_DOCS = [
    ("refund_policy",
     "Customers may request a full refund within 30 days of delivery. Refunds are "
     "issued to the original payment method within 5 to 7 business days."),
    ("return_policy",
     "To return an item, start a return from your orders page. Items must be unused "
     "and in original packaging. Return shipping is free for defective products."),
    ("shipping_policy",
     "Standard shipping takes 3 to 5 business days. Expedited shipping arrives in 1 "
     "to 2 business days. We ship to most countries; customs fees are the buyer's "
     "responsibility."),
    ("warranty",
     "All hardware includes a 1 year limited warranty covering manufacturing defects. "
     "The warranty does not cover accidental damage or normal wear."),
    ("sku_definition",
     "A SKU (stock keeping unit) is a unique identifier assigned to each distinct "
     "product and variant, used to track inventory across the catalog."),
]


def seed_sqlite(db_path: str) -> None:
    """Create and populate the demo `orders` table. Idempotent."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS orders;"
            "CREATE TABLE orders ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  customer TEXT NOT NULL,"
            "  amount REAL NOT NULL,"
            "  created_at TEXT NOT NULL"
            ");"
        )
        conn.executemany(
            "INSERT INTO orders (customer, amount, created_at) VALUES (?, ?, ?)",
            _ORDERS,
        )
        conn.commit()
    finally:
        conn.close()


async def seed_vector_store(store: InMemoryVectorStore) -> None:
    for doc_id, text in _DOCS:
        await store.add(doc_id=doc_id, text=text, source=doc_id)
