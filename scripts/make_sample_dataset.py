import csv
import random
from datetime import date, timedelta

from src.config import SAMPLE_DATASET_PATH

# fixed seed so the data, and the test answers, never change
RANDOM_SEED = 42
ROW_COUNT = 500

CITIES = ["Ramallah", "Nablus", "Hebron", "Bethlehem", "Jenin", "Jericho"]

PRODUCTS = {
    "Electronics": {"Laptop": 900, "Phone": 600, "Headphones": 80},
    "Clothing": {"Jacket": 70, "T-Shirt": 15, "Shoes": 55},
    "Home & Kitchen": {"Coffee Maker": 120, "Blender": 60, "Pan Set": 45},
    "Books": {"Novel": 12, "Cookbook": 25, "Dictionary": 30},
}

CUSTOMERS = ["أحمد خليل", "سارة يوسف", "محمد عودة", "ليلى حداد", "يوسف نصار", "رنا عيسى", "خالد سمارة", "نور شاهين"]

HEADER = ["Order ID", "Order Date", "City", "Category", "Product", "Quantity", "Unit Price ($)", "Total ($)",
          "Customer Name"]


def make_rows():
    rng = random.Random(RANDOM_SEED)
    start = date(2023, 1, 1)
    days = (date(2024, 12, 31) - start).days

    rows = []
    for _ in range(ROW_COUNT):
        order_date = start + timedelta(days=rng.randint(0, days))
        category = rng.choice(list(PRODUCTS))
        product = rng.choice(list(PRODUCTS[category]))
        price = PRODUCTS[category][product]
        quantity = rng.randint(1, 5)
        rows.append([order_date.isoformat(), rng.choice(CITIES), category, product,
                     quantity, price, quantity * price, rng.choice(CUSTOMERS)])

    rows.sort(key=lambda row: row[0])
    return [[order_id, *row] for order_id, row in enumerate(rows, start=1)]


if __name__ == "__main__":
    rows = make_rows()
    with open(SAMPLE_DATASET_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {SAMPLE_DATASET_PATH}")
