'''
Created by Qunkun Ma
pulls in 100000 entries from OpenAlex API where the authorship institution is U-M
'''

import pandas as pd
from pyalex import Works, config

# Configuration
#Kolbe API Key
api_key = 'NGABT8lQbr5z8VxZ9r5AHe'

config.api_key = api_key

umich_works_pager = Works().filter(authorships={"institutions": {"ror": "00jmfr291"}}).paginate(per_page=200, n_max=100000)

all_works = []
for page in umich_works_pager:
    all_works.extend(page)
    print(f"Collected so far: {len(all_works)}")


# Output to csv
df_100k = pd.DataFrame(all_works)
df_100k.to_csv("../data/raw/umich_works_100k".csv, index=False)
