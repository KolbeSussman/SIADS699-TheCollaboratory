'''
Created by Qunkun Ma
pulls in 100000 entries from OpenAlex API where the authorship institution is U-M

Note: will take ~30 minutes, depending on compute power/speed
'''

import pandas as pd
from pyalex import Works, config

# Configuration
#Kolbe API Key
api_key = 'insert your api key here'
config.api_key = api_key


OUTPUT_CSV = "data/raw/umich_works_100k.csv"


# Data Collection
umich_works_pager = Works().filter(authorships={"institutions": {"ror": "00jmfr291"}}).paginate(per_page=200, n_max=100000)

all_works = []
for page in umich_works_pager:
    all_works.extend(page)
    print(f"Collected so far: {len(all_works)}")


# Output to csv
if all_works:
    df_100k = pd.DataFrame(all_works)
    df_100k.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_100k)} records to {OUTPUT_CSV}")
else:
    print("No works collected. CSV not created.")

print("Data collection finished.")
