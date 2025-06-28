import os
import json
import random
import string

# Set your folder path
folder_path = "./"  # update to your actual path if needed

seen_ids = set()

def make_unique_id(existing_id):
    suffix = ''.join(random.choices(string.digits, k=4))
    return f"{existing_id}_{suffix}"

# Process each .json file
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        full_path = os.path.join(folder_path, filename)
        with open(full_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {filename}")
                continue

        modified = False
        for item in data:
            qid = item.get("query_id", "")
            if qid in seen_ids:
                new_id = make_unique_id(qid)
                print(f"🔁 Duplicate found in {filename}: {qid} → {new_id}")
                item["query_id"] = new_id
                modified = True
                seen_ids.add(new_id)
            else:
                seen_ids.add(qid)

        # Overwrite only if changes were made
        if modified:
            with open(full_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Updated: {filename}")
