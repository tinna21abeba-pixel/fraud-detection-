import pandas as pd
import numpy as np
import ipaddress

print('Reading CSVs')
# Use workspace-root-relative paths (notebooks used '../data/' because they run from notebooks/)
df = pd.read_csv('data/Fraud_Data .csv')
df_ip = pd.read_csv('data/IpAddress_to_Country.csv')

print('Columns in df:', list(df.columns))
print('Columns in df_ip:', list(df_ip.columns))


def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(str(ip).strip()))
    except Exception as e:
        return np.nan

print('Converting IPs...')
df['ip_int'] = df['ip_address'].apply(ip_to_int)
df_ip['lower_int'] = df_ip['lower_bound_ip_address'].apply(ip_to_int)
df_ip['upper_int'] = df_ip['upper_bound_ip_address'].apply(ip_to_int)

print('ip_int NaN count in df:', df['ip_int'].isna().sum())
print('Sample ip_address values:', df['ip_address'].head(10).tolist())

# Show unique types
print('dtype of ip_address in df:', df['ip_address'].dtype)

# Attempt to dropna and convert types
before = len(df)
df = df.dropna(subset=['ip_int'])
print(f'Dropped {before - len(df)} rows with invalid IPs')

try:
    df['ip_int'] = df['ip_int'].astype('int64')
    df_ip['lower_int'] = df_ip['lower_int'].astype('int64')
    df_ip['upper_int'] = df_ip['upper_int'].astype('int64')
    print('Converted to int64 successfully')
except Exception as e:
    print('Error converting to int64:', e)

# Sort and merge
try:
    df_sorted = df.sort_values('ip_int').reset_index(drop=True)
    df_ip_sorted = df_ip.sort_values('lower_int').reset_index(drop=True)
    merged = pd.merge_asof(df_sorted, df_ip_sorted[['lower_int','upper_int','country']], left_on='ip_int', right_on='lower_int', direction='backward')
    merged['country'] = np.where(merged['ip_int'] <= merged['upper_int'], merged['country'], 'Unknown')
    print('Merge successful. merged shape:', merged.shape)
    print('Top countries:', merged['country'].value_counts().head(10).to_dict())
except Exception as e:
    print('Error during merge_asof:', e)
    raise
