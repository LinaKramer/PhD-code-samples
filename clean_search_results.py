"""
@author: kramer

"""
import pandas as pd
import re
import numpy as np


# File paths
scopus_file_path = "data/cleaned_scopus_v1.csv"
wos_file_path = "data/cleaned_WoS_v1.xls"
econlit_file_path = "data/cleaned_EconLit_v1.xls"
jstor_file_path = "data/cleaned_JSTOR_v1.txt"

# Read in the data
df_scopus = pd.read_csv(scopus_file_path)
df_wos = pd.read_excel(wos_file_path)
df_econlit = pd.read_excel(econlit_file_path)

# Function to parse JSTOR file
def parse_jstor_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        entries = re.split(r'@article|@inbook', content)
        data = []
        for entry in entries[1:]:
            entry_dict = {}
            for line in entry.split(',\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    entry_dict[key.strip().capitalize()] = value.strip().strip('{}')
            data.append(entry_dict)
    df = pd.DataFrame(data)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].str.replace('}', '').str.strip()
    return df

df_jstor = parse_jstor_file(jstor_file_path)

# Function to clean titles
def clean_title(title):
    title = title.strip().lower()  # Strip whitespace and convert to lowercase
    title = re.sub(r'\s+', ' ', title)  # Replace multiple spaces with a single space
    title = re.sub(r'[^\w\s]', '', title)  # Remove special characters and punctuation
    return title

# Apply the cleaning function to the Title columns
df_scopus['cleaned_title'] = df_scopus['Title'].apply(clean_title)
df_wos['cleaned_title'] = df_wos['Title'].apply(clean_title)
df_econlit['cleaned_title'] = df_econlit['Title'].apply(clean_title)
df_jstor['cleaned_title'] = df_jstor['Title'].apply(clean_title)

# Check and delete duplicates within each dataset based on cleaned_title
df_scopus_duplicates = df_scopus[df_scopus.duplicated(subset='cleaned_title', keep=False)]
df_wos_duplicates = df_wos[df_wos.duplicated(subset='cleaned_title', keep=False)]
df_econlit_duplicates = df_econlit[df_econlit.duplicated(subset='cleaned_title', keep=False)]
df_jstor_duplicates = df_jstor[df_jstor.duplicated(subset='cleaned_title', keep=False)]

# Evaluate duplicates and delete manually based on author(s) and abstracts
# Manually clean files from unwanted document types (e.g. reviews, features, research reports, retracted articles and JSTOR results that do not provide an abstract)

# Check for duplicates by comparing the Title column of df_scopus and df_wos
# Begin the process for df_wos
duplicates_title = []

for index_scopus, row_scopus in df_scopus.iterrows():
    for index_wos, row_wos in df_wos.iterrows():
        if row_scopus['cleaned_title'] == row_wos['cleaned_title']:
            duplicates_title.append((index_scopus, index_wos))

# Remove duplicates from df_wos
df_wos = df_wos[~df_wos.index.isin([index_wos for _, index_wos in duplicates_title])]

# Repeat the process for df_scopus and df_econlit 
duplicates_title = []

for index_scopus, row_scopus in df_scopus.iterrows():
    for index_econlit, row_econlit in df_econlit.iterrows():
        if row_scopus['cleaned_title'] == row_econlit['cleaned_title']:
            duplicates_title.append((index_scopus, index_econlit))

# Remove duplicates from df_econlit
df_econlit = df_econlit[~df_econlit.index.isin([index_econlit for _, index_econlit in duplicates_title])]

# Repeat the process for df_wos and df_econlit
duplicates_title = []

for index_wos, row_wos in df_wos.iterrows():
    for index_econlit, row_econlit in df_econlit.iterrows():
        if row_wos['cleaned_title'] == row_econlit['cleaned_title']:
            duplicates_title.append((index_wos, index_econlit))

# No result

# Repeat the process for df_scopus and df_jstor
duplicates_title = []

for index_scopus, row_scopus in df_scopus.iterrows():
    for index_jstor, row_jstor in df_jstor.iterrows():
        if row_scopus['cleaned_title'] == row_jstor['cleaned_title']:
            duplicates_title.append((index_scopus, index_jstor))

# Remove duplicates from df_jstor based on Title
df_jstor = df_jstor[~df_jstor.index.isin([index_jstor for _, index_jstor in duplicates_title])]

# Repeat the process for df_wos and df_jstor
duplicates_title = []

for index_wos, row_wos in df_wos.iterrows():
    for index_jstor, row_jstor in df_jstor.iterrows():
        if row_wos['cleaned_title'] == row_jstor['cleaned_title']:
            duplicates_title.append((index_wos, index_jstor))

# No result

# Repeat the process for df_econlit and df_jstor
duplicates_title = []

for index_econlit, row_econlit in df_econlit.iterrows():
    for index_jstor, row_jstor in df_jstor.iterrows():
        if row_econlit['cleaned_title'] == row_jstor['cleaned_title']:
            duplicates_title.append((index_econlit, index_jstor))

# No result

# Save the final DataFrames as excel files
df_scopus.to_excel("data/cleaned_scopus_v2.xlsx", index=False)
df_wos.to_excel("data/cleaned_WoS_v2.xlsx", index=False)
df_econlit.to_excel("data/cleaned_EconLit_v2.xlsx", index=False)
df_jstor.to_excel("data/cleaned_JSTOR_v2.xlsx", index=False)

# Create df_merged_results with the specified columns
df_merged_results = pd.DataFrame(columns=[
    'Author(s)', 'Year', 'Title', 'Abstract', 'Keywords', 'Document type', 'Journal', 'Volume', 'Issue', 'Pages', 'Publisher', 'No. cited', 'Language of publication', 'URL', 'Source'
])

# Function to format author names
def format_authors(authors):
    return '; '.join([author.split(' (')[0] for author in authors.split('; ')])

# Match columns from df_scopus
df_scopus['Keywords'] = df_scopus.apply(lambda row: ', '.join(set(filter(None, (str(row['Author Keywords']) if pd.notna(row['Author Keywords']) else '').split(', ') + (str(row['Index Keywords']) if pd.notna(row['Index Keywords']) else '').split(', ')))), axis=1)
df_scopus['Author(s)'] = df_scopus['Author Full Names'].apply(format_authors)
df_scopus['Pages'] = df_scopus.apply(lambda row: f"{str(row['Page start'])} - {str(row['Page end'])}" if pd.notna(row['Page start']) and pd.notna(row['Page end']) else '', axis=1)
df_scopus_renamed = df_scopus.rename(columns={
    'Document Type': 'Document type',
    'Language of Original Document': 'Language of publication',
    'Title': 'Title',
    'Year': 'Year',
    'Abstract': 'Abstract',
    'Keywords': 'Keywords',
    'Source title': 'Journal',
    'Volume': 'Volume',
    'Issue': 'Issue',
    'Publisher': 'Publisher',
    'Cited by': 'No. cited',
    'Link': 'URL',
    'Source': 'Source'
})

# Select relevant columns
df_scopus_selected = df_scopus_renamed[[
    'Author(s)', 'Year', 'Title', 'Abstract', 'Keywords', 'Document type', 'Journal', 'Volume', 'Issue', 'Pages', 'Publisher', 'No. cited', 'Language of publication', 'URL', 'Source'
]]

# Append to df_merged_results
df_merged_results = pd.concat([df_merged_results, df_scopus_selected.reset_index(drop=True)], ignore_index=True)

# Match columns from df_wos
df_wos['Keywords'] = df_wos.apply(lambda row: ', '.join(set(filter(None, (str(row['Author Keywords']) if pd.notna(row['Author Keywords']) else '').split(', ') + (str(row['Keywords Plus']) if pd.notna(row['Keywords Plus']) else '').split(', ')))), axis=1)
df_wos['Author(s)'] = df_wos['Author Full Names'].apply(format_authors)
df_wos['Pages'] = df_wos.apply(lambda row: f"{str(row['Start Page'])} - {str(row['End Page'])}" if pd.notna(row['Start Page']) and pd.notna(row['End Page']) else '', axis=1)
df_wos_renamed = df_wos.rename(columns={
    'Document Type': 'Document type',
    'Language': 'Language of publication',
    'Title': 'Title',
    'Publication Year': 'Year',
    'Abstract': 'Abstract',
    'Keywords': 'Keywords',
    'Source Title': 'Journal',
    'Volume': 'Volume',
    'Issue': 'Issue',
    'Publisher': 'Publisher',
    'Times Cited, WoS Core': 'No. cited',
    'DOI Link': 'URL',
})
df_wos_renamed['Source'] = 'Web of Science'

# Rename document types in df_wos
df_wos_renamed['Document type'] = df_wos_renamed['Document type'].replace({
    'Proceedings Paper': 'Conference paper'
})

# Select relevant columns
df_wos_selected = df_wos_renamed[[
    'Author(s)', 'Year', 'Title', 'Abstract', 'Keywords', 'Document type', 'Journal', 'Volume', 'Issue', 'Pages', 'Publisher', 'No. cited', 'Language of publication', 'URL', 'Source'
]]

# Append to df_merged_results
df_merged_results = pd.concat([df_merged_results, df_wos_selected.reset_index(drop=True)], ignore_index=True)

# Match columns from df_econlit
df_econlit['Keywords'] = np.nan
df_econlit['Author(s)'] = df_econlit['Authors']
df_econlit['Pages'] = df_econlit['pages']
df_econlit['Publisher'] = np.nan
df_econlit_renamed = df_econlit.rename(columns={
    'documentType': 'Document type',
    'language': 'Language of publication',
    'Title': 'Title',
    'year': 'Year',
    'Abstract': 'Abstract',
    'Keywords': 'Keywords',
    'pubtitle': 'Journal',
    'volume': 'Volume',
    'issue': 'Issue',
    'DocumentURL': 'URL'
})
df_econlit_renamed['Source'] = 'EconLit'

# Rename document types in df_econlit
df_econlit_renamed['Document type'] = df_econlit_renamed['Document type'].replace({
    'Journal Article': 'Article',
    'Conference Proceedings': 'Conference paper'
})

# Select relevant columns
df_econlit_selected = df_econlit_renamed[[
    'Author(s)', 'Year', 'Title', 'Abstract', 'Keywords', 'Document type', 'Journal', 'Volume', 'Issue', 'Pages', 'Publisher', 'Language of publication', 'URL', 'Source'
]]

# Append to df_merged_results
df_merged_results = pd.concat([df_merged_results, df_econlit_selected.reset_index(drop=True)], ignore_index=True)

# Function to format JSTOR author names
def format_jstor_authors(authors):
    if pd.isna(authors):
        return ''
    return '; '.join([', '.join(author.split(' ')[::-1]) for author in authors.split(' and ')])

# Set default values for missing columns in df_jstor
df_jstor['Keywords'] = np.nan
df_jstor['Language of publication'] = 'English'
df_jstor['Document type'] = df_jstor.apply(lambda row: 'Book chapter' if pd.notna(row.get('Booktitle')) else 'Article', axis=1)

# Match columns from df_jstor
df_jstor['Author(s)'] = df_jstor['Author'].apply(format_jstor_authors)
df_jstor['Pages'] = df_jstor['Pages'].str.replace('--', '-')
df_jstor_renamed = df_jstor.rename(columns={
    'Title': 'Title',
    'Year': 'Year',
    'Abstract': 'Abstract',
    'Journal': 'Journal',
    'Volume': 'Volume',
    'Number': 'Issue',
    'Publisher': 'Publisher',
    'Url': 'URL',
})
df_jstor_renamed['Source'] = 'JSTOR'

# Select relevant columns
df_jstor_selected = df_jstor_renamed[[
    'Author(s)', 'Year', 'Title', 'Abstract', 'Keywords', 'Document type', 'Journal', 'Volume', 'Issue', 'Pages', 'Publisher', 'Language of publication', 'URL', 'Source'
]]

# Append to df_merged_results
df_merged_results = pd.concat([df_merged_results, df_jstor_selected.reset_index(drop=True)], ignore_index=True)

# Remove text after the © sign in the Abstract column
df_merged_results['Abstract'] = df_merged_results['Abstract'].apply(lambda x: re.split(r'©', x)[0] if pd.notna(x) else x)

# Save df_merged_results as an Excel file
df_merged_results.to_excel("data/merged_search_results_v1.xlsx", index=False)
