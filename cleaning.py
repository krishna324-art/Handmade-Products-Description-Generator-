import json
import re

input_file = r"C:\Users\Abc\Downloads\meta_Handmade_Products.jsonl"
output_file = "meta_Handmade_Products_cleaned.jsonl"
MODE = 'random'

kept_count = 0
removed_count = 0

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            item = json.loads(line)

            desc = item.get("description")
            features = item.get("features")

            # Process description: if list, join to string
            if isinstance(desc, list):
                desc = " ".join(str(d) for d in desc).strip()

            # Check description validity
            if not desc:
                removed_count += 1
                continue
            if len(desc) < 20 or desc.lower() in ["n/a", "none", "no description"]:
                removed_count += 1
                continue
            if "http" in desc or re.search(r"<.*?>", desc):
                removed_count += 1
                continue

            # Normalize whitespace in description
            desc = re.sub(r"\s+", " ", desc).strip()
            item["description"] = desc  # update cleaned description

            # Check features validity (must be a non-empty list)
            if not features or not (isinstance(features, list) and len(features) > 0):
                removed_count += 1
                continue

            # Remove unwanted fields
            item.pop("average_rating", None)
            item.pop("rating_number", None)
            item.pop("details", None)
            item.pop("parent_asin", None)
            item.pop("bought_together", None)

            # Remove images and videos
            item.pop("images", None)
            item.pop("videos", None)

            outfile.write(json.dumps(item) + "\n")
            kept_count += 1

        except json.JSONDecodeError:
            removed_count += 1
            continue

print(f"✅ Cleaning complete!")
print(f"Kept {kept_count} records")
print(f"Removed {removed_count} records (empty description or features or bad data)")
print(f"Saved to {output_file}")

# Viewing first 10 lines of the cleaned output file (pretty printed)
print("\n📄 First 10 lines of cleaned output:")
with open(output_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        item = json.loads(line)
        print(json.dumps(item, indent=4, ensure_ascii=False))
        print("-" * 40)  # separator between records

with open(output_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines in dataset: {total_lines}")

