"""
@author: kramer
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_merged_results_path = "data/merged_search_results_v5.xlsx"

df_merged_results = pd.read_excel(df_merged_results_path)

# Check answers 
research_focus_check_df = df_merged_results[['Category of Research Focus', 'Category of Research Focus Refinement', 'Subcategory of Research Focus' ]]
ML_focus_check_df = df_merged_results[['Type of ML', 'Type of ML Refinement', 'Specific Technique']]

# Remove "(EG)" from all cells
df_merged_results = df_merged_results.replace(r' \(EG\)', '', regex=True)


""" Bibliometric Analysis:""" 
# Publication Source:
source = df_merged_results['Source'].value_counts()
print("\nSources:\n", source)

# Publication Trends Over Time:
years = df_merged_results['Year'].value_counts()
print("\nYears:\n", years)

plt.figure(figsize=(12, 8))
df_merged_results['Year'] = pd.to_numeric(df_merged_results['Year'], errors='coerce')
df_merged_results = df_merged_results.dropna(subset=['Year'])
df_merged_results['Year'] = df_merged_results['Year'].astype(int)
df_merged_results = df_merged_results.sort_values(by='Year')
sns.countplot(x='Year', data=df_merged_results, order=df_merged_results['Year'].sort_values().unique())
plt.title('Publication Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Publication Type and Source:
# Document type
document_types = df_merged_results['Document type'].value_counts()
print("\nDocument Types:\n", document_types)

# Publisher
publishers = df_merged_results['Publisher'].value_counts()
# Count IEEE, Springer, Elsevier & ACM
ieee_publisher_count = df_merged_results['Publisher'].str.contains('IEEE|Institute of Electrical and Electronics Engineers', case=False, na=False).sum()
acm_publisher_count = df_merged_results['Publisher'].str.contains('ACM|Association for Computing Machinery', case=False, na=False).sum()
springer_publisher_count = df_merged_results['Publisher'].str.contains('Springer', case=False, na=False).sum()
elsevier_publisher_count = df_merged_results['Publisher'].str.contains('Elsevier', case=False, na=False).sum()

print("\nPublishers:\n", publishers)
print(f"Number of publishers containing 'IEEE': {ieee_publisher_count}")
print(f"Number of publishers containing 'ACM': {acm_publisher_count}")
print(f"Number of publishers containing 'springer': {springer_publisher_count}")
print(f"Number of publishers containing 'ELsevier': {elsevier_publisher_count}")

# Filter and count document types for each publisher category
ieee_docs = df_merged_results[df_merged_results['Publisher'].str.contains('IEEE|Institute of Electrical and Electronics Engineers', case=False, na=False)]
acm_docs = df_merged_results[df_merged_results['Publisher'].str.contains('ACM|Association for Computing Machinery', case=False, na=False)]
springer_docs = df_merged_results[df_merged_results['Publisher'].str.contains('Springer', case=False, na=False)]
elsevier_docs = df_merged_results[df_merged_results['Publisher'].str.contains('Elsevier', case=False, na=False)]

# Count document types for each category
ieee_doc_types = ieee_docs['Document type'].value_counts()
acm_doc_types = acm_docs['Document type'].value_counts()
springer_doc_types = springer_docs['Document type'].value_counts()
elsevier_doc_types = elsevier_docs['Document type'].value_counts()

# Print the results
print("IEEE Document Types:\n", ieee_doc_types)
print("\nACM Document Types:\n", acm_doc_types)
print("\nSpringer Document Types:\n", springer_doc_types)
print("\nElsevier Document Types:\n", elsevier_doc_types)


# Journal:
journals = df_merged_results['Journal'].value_counts()
# Count IEEE & ACM
ieee_journal_count = df_merged_results['Journal'].str.contains('IEEE', case=False, na=False).sum()
acm_journal_count = df_merged_results['Journal'].str.contains('ACM', case=False, na=False).sum()

print("\nJournals:\n", journals)
print(f"Number of journals containing 'IEEE': {ieee_journal_count}")
print(f"Number of journals containing 'ACM': {acm_journal_count}")


# Language:
df_merged_results['Language of publication'] = df_merged_results['Language of publication'].str.strip().replace(" English", "English")
languages = df_merged_results['Language of publication'].value_counts()
print("\nLanguages:\n", languages)


# Document length:
def calculate_page_length(pages):
    if pd.isna(pages):
        return None
    try:
        start_page, end_page = pages.split(' - ')
        start_page = int(start_page.split('.')[0])  # Fix the formatting issue
        end_page = int(end_page)
        return end_page - start_page
    except ValueError:
        return None

# Apply the function to calculate the length of each publication
df_merged_results['Page Length'] = df_merged_results['Pages'].apply(calculate_page_length)
page_lengths = df_merged_results['Page Length'].describe()

print("\nPage Lengths:\n", page_lengths)

# Author(s):
from pyvis.network import Network
import networkx as nx

# Split the Authors into lists
df_merged_results['Authors List'] = df_merged_results['Author(s)'].apply(lambda x: x.split('; ') if pd.notna(x) else [])

# Flatten the lists and count each author and keyword
all_authors = [author for sublist in df_merged_results['Authors List'] for author in sublist]
author_counts = pd.Series(all_authors).value_counts()
print("Author Counts:\n", author_counts)

# Function to create the co-authorship network
def create_coauthorship_network(df):
    coauthorship_network = nx.Graph()
    for authors in df['Authors List']:
        for i, author1 in enumerate(authors):
            for author2 in authors[i + 1:]:
                if coauthorship_network.has_edge(author1, author2):
                    coauthorship_network[author1][author2]['weight'] += 1
                else:
                    coauthorship_network.add_edge(author1, author2, weight=1)
    return coauthorship_network

# Filter authors who have written multiple papers
author_counts = df_merged_results['Authors List'].explode().value_counts()
multiple_paper_authors = author_counts[author_counts > 1].index

# Filter the DataFrame to include only these authors
df_filtered = df_merged_results.copy()
df_filtered['Authors List'] = df_filtered['Authors List'].apply(lambda authors: [author for author in authors if author in multiple_paper_authors])
df_filtered = df_filtered[df_filtered['Authors List'].map(len) > 0]

# Create the co-authorship network
coauthorship_network = create_coauthorship_network(df_filtered)


# Create a pyvis network
net = Network(notebook=True, height="750px", width="100%", bgcolor="white", font_color="black")

# Convert the networkx graph to pyvis
net.from_nx(coauthorship_network)
print(net)

# Customize the appearance
net.repulsion(node_distance=100, central_gravity=0.3, spring_length=100, spring_strength=0.05, damping=0.09)

# Increase the font size of the labels
for node in net.nodes:
    node['font'] = {'size': 20}  

# Show the network with interactive buttons
net.show_buttons(filter_=['physics'])
net.show("coauthorship_network.html")


# Citation Trends
citation_counts = df_merged_results['No. cited'].describe()
print("\nCitation Counts:\n", citation_counts)

# Citations per Year:
# Create a copy of the DataFrame for calculations
df_citations = df_merged_results.copy()

# Ensure the 'Year' column is numeric
df_citations['Year'] = pd.to_numeric(df_citations['Year'], errors='coerce')

# Drop rows with NaN values in 'Year' or 'No. cited'
df_citations = df_citations.dropna(subset=['Year', 'No. cited'])

# Convert 'Year' to integer
df_citations['Year'] = df_citations['Year'].astype(int)

# Calculate the current year
current_year = pd.Timestamp.now().year

# Calculate the number of years since publication
df_citations['Years Since Publication'] = current_year - df_citations['Year']

# Calculate citations per year
df_citations['Citations Per Year'] = df_citations['No. cited'] / df_citations['Years Since Publication']

# Group by 'Year' and calculate the total number of citations and the number of papers
citations_per_year = df_citations.groupby('Year')['No. cited'].sum().reset_index()
papers_per_year = df_citations.groupby('Year').size().reset_index(name='Number of Papers')

# Merge the two DataFrames on 'Year'
citations_and_papers_per_year = pd.merge(citations_per_year, papers_per_year, on='Year')

# Calculate the average number of citations per paper for each year
citations_and_papers_per_year['Average Citations Per Paper'] = citations_and_papers_per_year['No. cited'] / citations_and_papers_per_year['Number of Papers']

# Plot the graph for total citations per year
plt.figure(figsize=(12, 8))
plt.plot(citations_per_year['Year'], citations_per_year['No. cited'], marker='o')
plt.title('Total Citations Per Year')
plt.xlabel('Year')
plt.ylabel('Total Citations')
plt.xticks(citations_per_year['Year'], rotation=45)  # Set x-ticks to be the unique years
plt.grid(True)
plt.show()

# Plot the graph for average citations per paper per year
plt.figure(figsize=(12, 8))
plt.plot(citations_and_papers_per_year['Year'], citations_and_papers_per_year['Average Citations Per Paper'], marker='o')
plt.title('Average Citations Per Year')
plt.xlabel('Year')
plt.ylabel('Average Citations Per Paper')
plt.xticks(citations_and_papers_per_year['Year'], rotation=45)  # Set x-ticks to be the unique years
plt.grid(True)
plt.show()




# Keyword co-occurence analysis:
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download the WordNet data and Punkt tokenizer models
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# Function to clean and normalize keywords
def clean_keyword(keyword):
    keyword = keyword.lower()  # Convert to lowercase
    keyword = re.sub(r'\(.*?\)', '', keyword)  # Remove parentheses and content within them
    keyword = re.sub(r'\s+', ' ', keyword)  
    keyword = keyword.replace(".", "")    
    keyword = keyword.replace("-", " ")
    keyword = keyword.replace("'", "")
    keyword = keyword.replace("taxation", "tax")
    keyword = keyword.strip()
    words = word_tokenize(keyword)  # Tokenize the keyword into words
    lemmas = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize each word
    return ' '.join(lemmas)  # Join the lemmatized words back into a single string


# Function to check if two words are similar by one letter difference
def are_one_letter_different(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio() >= 0.9

# Function to remove similar words from a list based on one-letter difference
def remove_similar_words(keywords_list):
    unique_keywords = []
    similar_pairs_one_letter = []
    
    for keyword in keywords_list:
        found_similar = False
        for unique_keyword in unique_keywords:
            if are_one_letter_different(keyword, unique_keyword):
                similar_pairs_one_letter.append((keyword, unique_keyword))
                found_similar = True
                break
        if not found_similar:
            unique_keywords.append(keyword)
    
    return unique_keywords, similar_pairs_one_letter

# Split the keywords, clean them, and flatten the list
df_merged_results['Keywords List'] = df_merged_results['Keywords'].str.split(';|,').apply(lambda x: [clean_keyword(keyword) for keyword in x] if isinstance(x, list) else [])

# Process each row in the 'Keywords List' column
df_merged_results['Cleaned Keywords List'], df_merged_results['Similar Words One Letter'] = zip(*df_merged_results['Keywords List'].apply(remove_similar_words))

all_keywords = df_merged_results['Cleaned Keywords List'].explode().dropna().str.strip()

# Count the most common keywords in all_keywords
most_common_keywords = all_keywords.value_counts()

unique_keywords = all_keywords.unique().tolist()

# Print the keywords that were removed and their similar counterparts
removed_keywords_one_letter = df_merged_results['Similar Words One Letter'].explode().dropna().str.strip()

# For manual control:
print("\nRemoved Keywords (One Letter Difference) and their similar counterparts:")
for pair in df_merged_results['Similar Words One Letter'].explode().dropna():
    print(pair)

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_keywords))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
#plt.title('Word Cloud of Keywords')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from gensim.models import KeyedVectors
from scipy.spatial import ConvexHull


# Create a document-term matrix directly from the list of keywords
vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
dtm = vectorizer.fit_transform(df_merged_results['Cleaned Keywords List'])

# Convert the document-term matrix to a DataFrame
dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())


# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(dtm_df)

# Add the PCA coordinates to the DataFrame
df_merged_results['PCA1'] = pca_coords[:, 0]
df_merged_results['PCA2'] = pca_coords[:, 1]

# Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(pca_coords)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Perform KMeans clustering with the chosen number of clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pca_coords)
df_merged_results['Cluster'] = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Find the publication closest to the center of each cluster
closest_publications__keyword = {}
closest_publications__author = {}

for cluster in range(num_clusters):
    cluster_indices = np.where(df_merged_results['Cluster'] == cluster)[0]
    cluster_pca_coords = pca_coords[cluster_indices]
    cluster_center = cluster_centers[cluster]
    
    # Calculate the Euclidean distance between each publication's PCA coordinates and the cluster center
    distances = np.linalg.norm(cluster_pca_coords - cluster_center, axis=1)
    
    # Find the index of the publication with the smallest distance to the cluster center
    closest_publication_index = cluster_indices[np.argmin(distances)]
    
    # Get the keywords corresponding to the closest publication index
    closest_publication_keywords = df_merged_results.iloc[closest_publication_index]['Cleaned Keywords List']
    
    closest_publications__keyword[cluster] = closest_publication_keywords
    
    # Get the author and year information
    author = df_merged_results.iloc[closest_publication_index]['Author(s)']
    year = df_merged_results.iloc[closest_publication_index]['Year']
    if len(author.split(',')) > 1:
        author = author.split(',')[0] + ' et al'
    closest_publications__author[cluster] = f"{author} ({year})"
    
    
# Print the closest publication keywords to the center of each cluster
print(closest_publications__keyword)
print(closest_publications__author)

# Analyze the keywords for each cluster
cluster_keywords = {}

for cluster in range(num_clusters):
    cluster_indices = np.where(df_merged_results['Cluster'] == cluster)[0]
    cluster_dtm = dtm_df.iloc[cluster_indices]
    
    # Calculate the frequency of each keyword in the cluster
    keyword_freq = cluster_dtm.sum(axis=0)
    
    # Calculate the overall frequency of each keyword
    overall_freq = dtm_df.sum(axis=0)
    
    # Calculate the relative frequency of each keyword in the cluster compared to the overall frequency
    relative_freq = (keyword_freq / overall_freq).sort_values(ascending=False)
    
    # Get the top 10 indicative keywords for the cluster
    top_keywords = relative_freq.head(50)
    cluster_keywords[cluster] = top_keywords

# Print the indicative keywords for each cluster
for cluster, keywords in cluster_keywords.items():
    print(f"Cluster {cluster}:")
    print(keywords)
    print()

# Cluster descriptions based on manual evaluation:
cluster_descriptions = {
    0: "Tax Policy Implementation & Behavioral Compliance",
    1: "Data-Driven Financial Forecasting & Decision-Making",
    2: "Sustainability, Eco-Taxation & Tax Policy Optimization",
    3: "Digital Economy, Cybersecurity & Tax Administration",
    4: "Legal Compliance, Fraud Detection & Risk Analysis"
}


# Plot the PCA results with clustering
plt.figure(figsize=(18, 10))
scatter = plt.scatter(df_merged_results['PCA1'], df_merged_results['PCA2'], c=df_merged_results['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Publications Based on Keywords', fontsize=14)
plt.xlabel('PCA Component 1', fontsize= 12)
plt.ylabel('PCA Component 2', fontsize =12)
plt.grid(True)

# Add center nodes and connect to outermost nodes
colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
for cluster in range(num_clusters):
    cluster_indices = np.where(df_merged_results['Cluster'] == cluster)[0]
    cluster_pca_coords = pca_coords[cluster_indices]
    cluster_center = cluster_centers[cluster]
    
    # Plot the center node
    plt.scatter(cluster_center[0], cluster_center[1], c=[colors[cluster]], marker='x', s=100)
    plt.annotate(closest_publications__author[cluster], (cluster_center[0], cluster_center[1]), fontsize=12, ha='right')
    
    # Identify the outermost nodes using ConvexHull
    if len(cluster_pca_coords) > 2:  # ConvexHull requires at least 3 points
        hull = ConvexHull(cluster_pca_coords)
        for simplex in hull.simplices:
            plt.plot(cluster_pca_coords[simplex, 0], cluster_pca_coords[simplex, 1], color=colors[cluster])
            for vertex in simplex:
                plt.plot([cluster_center[0], cluster_pca_coords[vertex, 0]], [cluster_center[1], cluster_pca_coords[vertex, 1]], color=colors[cluster], linestyle=' ')
        
        # Fill the area of the cluster with a transparent color
        hull_points = cluster_pca_coords[hull.vertices]
        plt.fill(hull_points[:, 0], hull_points[:, 1], color=colors[cluster], alpha=0.2)

# Create custom legend
handles, _ = scatter.legend_elements()
# Reorder the labels according to the specified order
ordered_labels = [cluster_descriptions[i] for i in [0,1,2,3,4]]
plt.legend(handles, ordered_labels, title="Clusters:", fontsize=12, title_fontsize=12)
plt.show()


# Analyze the impact of keywords on PCA components
pca_components = pd.DataFrame(pca.components_, columns=vectorizer.get_feature_names_out())
pca_components = pca_components.T
pca_components.columns = ['PCA1', 'PCA2']

# Find the keywords with the highest impact on PCA1 and PCA2
top_keywords_pca1 = pca_components['PCA1'].abs().sort_values(ascending=False).head(10)
top_keywords_pca2 = pca_components['PCA2'].abs().sort_values(ascending=False).head(10)

print("Top keywords impacting PCA1:")
print(top_keywords_pca1)

print("\nTop keywords impacting PCA2:")
print(top_keywords_pca2)




#########################################################################################################




""" Content Analysis:""" 
# Category of Research Focus:
category_of_research_focus = df_merged_results['Category of Research Focus Refinement'].value_counts()
category_of_research_focus_percentages = (category_of_research_focus / category_of_research_focus.sum()) * 100

print("\nCategory of Research Focus:\n", category_of_research_focus)
print("\nCategory of Research Focus Percentages:\n", category_of_research_focus_percentages)


main_category_year = df_merged_results.groupby('Year')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())
print("\nMain 'ML type' for each year:\n", main_category_year)

import matplotlib.pyplot as plt
import pandas as pd

# Ensure the 'Year' column is numeric
df_merged_results['Year'] = pd.to_numeric(df_merged_results['Year'], errors='coerce')

# Drop rows with NaN values in 'Year' or 'Category of Research Focus Refinement'
df_merged_results = df_merged_results.dropna(subset=['Year', 'Category of Research Focus Refinement'])

# Group by 'Year' and 'Category of Research Focus Refinement' and count the occurrences
category_counts = df_merged_results.groupby(['Year', 'Category of Research Focus Refinement']).size().unstack(fill_value=0)

# Plot the stacked area chart
category_counts.plot(kind='area', stacked=True, figsize=(14, 8), cmap='viridis')

# Customize the plot
plt.title('Research Focus by Year')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.legend(title='Category of Research Focus', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Ensure x-axis ticks are full integers
plt.xticks(ticks=range(int(df_merged_results['Year'].min()), int(df_merged_results['Year'].max()) + 1))

# Set x-axis limits to match the range of the data
plt.xlim(int(df_merged_results['Year'].min()), int(df_merged_results['Year'].max()))

# Show the plot
plt.show()


# Subcategory of Research Focus:
subcategory_of_research_focus = df_merged_results['Subcategory of Research Focus'].value_counts()
print("\nSubcategory of Research Focus:\n", subcategory_of_research_focus)

def categorize_research_subcategories(subcategory):
    subcategory = subcategory.lower()
    
    # Tax fraud detection
    if ('tax fraud detection' in subcategory or 
        'tax evasion' in subcategory or 
        'customs fraud' in subcategory):
        return 'Tax Fraud Detection'
    
    # Tax Processing
    if ('information retrieval' in subcategory or 
        'document management' in subcategory or 
        'tax record digitization' in subcategory or 
        'tax assessment automation' in subcategory or 
        'automated toll collection' in subcategory or 
        'tax procedure optimization' in subcategory or 
        'tax data utilization' in subcategory or
        'vat invoice processing and accuracy improvement' in subcategory or
        'decentralized taxation and redistribution' in subcategory or
        'entity recognition' in subcategory):
        return 'Tax Processing'
    
    # Tax Policy Evaluation
    if ('policy evaluation' in subcategory or 
        'economic policy' in subcategory or 
        'program evaluation' in subcategory or
        'tax policy impact' in subcategory or 
        'tax policy popularity' in subcategory or 
        'tax rate optimization' in subcategory or 
        'effectiveness of taxation' in subcategory or 
        'public acceptance' in subcategory or 
        'wealth redistribution' in subcategory or
        'evaluation of tax incentives' in subcategory or
        'allocation and distribution' in subcategory or
        'economic impact analysis' in subcategory or
        'impact of tax policy on market behavior' in subcategory or
        'impact of tax credits on household behavior' in subcategory or
        'impact of tax policy on investment decisions' in subcategory):
        return 'Tax Policy Evaluation'
    
    # Tax Policy Optimization
    if ('tax system optimization, information management' in subcategory or 
        'health insurance policy design' in subcategory or
        'resource allocation and optimization' in subcategory or
        'media influence on tax policy design' in subcategory or 
        'optimal taxation strategies' in subcategory):
        return 'Tax Policy Optimization'
    
    # Tax Revenue Forecasting
    if ('revenue forecasting' in subcategory or 
        'tax revenue prediction' in subcategory or 
        'tax forecasting' in subcategory or 
        'sales tax forecasting' in subcategory or 
        'revenue trend analysis' in subcategory or 
        'fiscal forecasting' in subcategory or
        'prediction of effective tax rates' in subcategory or
        'tax prediction' in subcategory or
        'tax debt' in subcategory or
        'income tax forecasting' in subcategory):
        return 'Tax Revenue Forecasting'
    
    # Public Opinion Analysis
    if ('public opinion' in subcategory or 
        'public sentiment' in subcategory or 
        'taxpayer behavior' in subcategory or 
        'behavioral response' in subcategory or 
        'influence of religion' in subcategory or 
        'tax morale' in subcategory or 
        'sentiment analysis' in subcategory or
        'public attitudes towards tax policy' in subcategory or
        'taxpayer response to tax policy' in subcategory or
        'vat acceptance analysis' in subcategory or
        'trust and compliance' in subcategory):
        return 'Public Opinion Analysis'
    
    # Corporate Taxation and Avoidance
    if ('corporate tax' in subcategory or 
        'tax avoidance' in subcategory or 
        'corporate behavior' in subcategory or 
        'tax planning' in subcategory or 
        'r&d manipulation' in subcategory or
        'tax risk identification' in subcategory or
        'tax risk management' in subcategory or
        'debt management analysis' in subcategory or
        'valuation of deferred taxes' in subcategory):
        return 'Corporate Taxation and Avoidance'
    
    # Property Tax Assessment
    if ('property tax' in subcategory or 
        'real estate tax' in subcategory or 
        'property valuations' in subcategory):
        return 'Property Tax Assessment'
    
    # Environmental Taxation
    if ('environmental taxation' in subcategory or 
        'eco-taxation' in subcategory or
        'environmental tax reform assessment' in subcategory or
        'environmental tax policy design' in subcategory or
        'tax climate analysis' in subcategory or
        'carbon tax' in subcategory):
        return 'Environmental Taxation'
    
    # Tax Law and Regulations
    if ('legal' in subcategory or 
        'statutory interpretation' in subcategory or 
        'tax compliance and regulation' in subcategory or
        'treaty formation' in subcategory):
        return 'Tax Law and Regulations'
    
    # Tax compliance 
    if ('tax compliance' in subcategory or 
        'taxpayer noncompliance' in subcategory or 
        'audit selection' in subcategory or 
        'risk assessment' in subcategory or 
        'automated tax compliance' in subcategory or
        'default prediction' in subcategory):
        return 'Tax Compliance'
    

    else: 
        return 'Not specified'


# Apply the categorization function
df_merged_results['Subcategory of Research Focus Categorized'] = df_merged_results['Subcategory of Research Focus'].apply(categorize_research_subcategories)

# Filter for entries categorized as "Other"
other_entries = df_merged_results[df_merged_results['Subcategory of Research Focus Categorized'] == 'Not specified']
other_entries = other_entries['Subcategory of Research Focus']

# Calculate the value counts for each category
subcategory_counts = df_merged_results['Subcategory of Research Focus Categorized'].value_counts()

# Calculate the percentage of each category
subcategory_counts_percentages = (subcategory_counts / subcategory_counts.sum()) * 100

# Print the counts and percentages
print("\nSubcategory of Research Focus Counts:\n", subcategory_counts)
print("\Subcategory of Research Focus Tax Percentages:\n", subcategory_counts_percentages)


# Group by 'Category of Research Focus Refinement' and 'Subcategory of Research Focus Categorized'
subcategory_counts = df_merged_results.groupby(['Category of Research Focus Refinement', 'Subcategory of Research Focus Categorized']).size().unstack(fill_value=0)

# Print the counts
print("\nOccurrences of each 'Subcategory of Research Focus Categorized' within each 'Category of Research Focus Refinement':\n")
print(subcategory_counts)



# Type of Taxes:
type_of_tax = df_merged_results['Type of Tax'].value_counts()
print("\nType of Tax:\n", type_of_tax)

# Replace "Taxes" with "Tax" in the 'Type of Tax' column
df_merged_results['Type of Tax'] = df_merged_results['Type of Tax'].str.replace('Taxes', 'Tax')

# Count occurrences of specific terms in 'Type of Tax'
tax_terms = [
    'Tax on individual income', 'Tax on individual wealth', 'Tax on individual consumption',
    'Tax on business income', 'Tax on trade'
]

def count_tax_terms(tax_string):
    counts = {term: 0 for term in tax_terms}
    for term in tax_terms:
        if term.lower() in tax_string.lower():
            counts[term] += 1
    return counts

tax_counts = df_merged_results['Type of Tax'].apply(count_tax_terms)
tax_counts_df = pd.DataFrame(list(tax_counts)).sum()
type_of_tax = df_merged_results['Type of Tax'].value_counts()
print("\nType of Tax:\n", type_of_tax)
print("\nTax Term Counts:\n", tax_counts_df)

# Calculate the percentage of each type of tax term
tax_counts_df_percentages = (tax_counts_df / tax_counts_df.sum()) * 100
print("\nTax Term Counts Percentages:\n", tax_counts_df_percentages)

# Plot Type of Tax per year
# Ensure the 'Year' column is numeric
df_merged_results['Year'] = pd.to_numeric(df_merged_results['Year'], errors='coerce')

# Define the tax terms
tax_terms = [
    'Tax on individual income', 'Tax on individual wealth', 'Tax on individual consumption',
    'Tax on business income', 'Tax on trade'
]

# Function to split rows with multiple tax types into separate rows
def split_multiple_tax_types(df):
    new_rows = []
    for _, row in df.iterrows():
        tax_types = row['Type of Tax'].split(', ')
        for tax_type in tax_types:
            new_row = row.copy()
            new_row['Type of Tax'] = tax_type
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)

# Apply the function to split rows with multiple tax types
df_split = split_multiple_tax_types(df_merged_results)

# Filter the DataFrame to include only the specified tax terms
df_split = df_split[df_split['Type of Tax'].isin(tax_terms)]

# Group by 'Year' and 'Type of Tax' and count the occurrences
tax_year_counts = df_split.groupby(['Year', 'Type of Tax']).size().reset_index(name='Count')

# Create the bubble chart
plt.figure(figsize=(14, 8))
bubble_chart = sns.scatterplot(
    data=tax_year_counts,
    x='Year',
    y='Type of Tax',
    size='Count',
    sizes=(20, 2000),
    legend=False,
    color='#8FBC8F',  
    alpha=0.6
)

# Customize the plot
# plt.title('Tax Types Investigated Over the Years')
plt.xlabel('Year')
plt.ylabel('Type of Tax')
plt.grid(True)

# Ensure y-axis is sorted according to tax_terms
bubble_chart.set_yticks(range(len(tax_terms)))
bubble_chart.set_yticklabels(tax_terms)

# Ensure x-axis ticks are full integers
plt.xticks(ticks=range(int(df_split['Year'].min()), int(df_split['Year'].max()) + 1))

# Add a legend for bubble sizes
scaling_factor = 2000 / max(tax_year_counts['Count'])  # Calculate the scaling factor
for size in [1, 10, 20, 30]:  
    plt.scatter([], [], s=size * scaling_factor, c='#8FBC8F', alpha=0.6, label=f'{size} counts')
plt.legend(scatterpoints=1, frameon=False, labelspacing=4, handletextpad=2, title='Bubble Size', loc='center left', bbox_to_anchor=(1.05, 0.5), facecolor='white')

# Show the plot
plt.show()


# Exact Taxes:
exact_tax = df_merged_results['Exact Tax'].value_counts()
print("\nExact Tax:\n", exact_tax)
    
# Define the categorization function
def categorize_exact_tax(tax):
    tax = tax.lower()
    if tax in ['personal income tax', 'individual income tax', 'income tax (individual)', 'progressive income taxes', 'progressive income tax', 'u.s. individual income tax', 'annual individual income tax return', 'labor tax', 'payroll tax', 'tax on workforce', 'income tax', 'general income tax', 'income tax returns', 'income tax compliance', 'income tax declarations', 'income tax evasion', 'income tax related claims by limited liability companies', 'general income tax', 'german income tax', 'north american income-tax forms', 'effective tax rates (etrs)', 'taxationâ€“redistribution mechanism', 'tax on individual income declarations', 'personal income tax returns (irpf in spanish)', 'annual individual income tax return (aiitr)']:
        return 'Individual Income Tax'
    elif tax in ['corporate income tax', 'corporate tax', 'business income tax', 'general business tax revenue', 'tax law cases, likely involving various business-related taxes','enterprise income tax', 'tax arrears in enterprises', 'tax on corporate profits', 'commercial taxes', 'sole proprietorship tax returns', 'commercial and industrial profits tax', 'corporate taxation', 'business tax', 'enterprise tax', 'commercial tax', 'corporate tax compliance', 'internal revenue code related to business taxation', 'tax compliance risk characteristics for smes', 'taxation data (specific tax type not mentioned, assumed to be related to business income due to b2b context)', 'corporate tax evasion', 'corporate tax revenue', 'corporate tax avoidance', 'tax on business income', 'tax on pass-through entities such as partnerships, trusts, and s-corporations', 'tax on corporate profits', 'tax on business income', 'tax on business transactions', 'tax on business income (eg - based on the context of firms and tax risk)', 'tax evasion related to business income', 'tax avoidance by government-link company in malaysia', 'tax avoidance cases among companies', 'deferred taxes', 'wash-sale tax', 'taxable amount declared by businesses', 'tax related to business transactions','tax liability insurance premium', 'taxes on microentrepreneurs','tax fraud detection, tax avoidance, tax compliance', 'corporate tax arrears', 'smes business income tax', 'business transaction taxes', 'business tax debts', 'enterprise taxpayer risk profile', 'business taxes under a standard tax regime', 'business tax risk identification', 'enterprise tax arrears', 'tax risk assessment (eg - likely related to business income tax as it involves revenue agency officers)']:
        return 'Corporate Income Tax'
    elif tax in ['digital tax', 'crypto tax', 'tax on digital platforms', 'tax on crypto gains', 'tax of 30% on crypto gains', 'digital economy income tax', 'taxes on digital platforms']:
        return 'Digital & Tech Tax'
    elif tax in ['property tax (specifically business rates)', 'swedish wealth tax of 1571']:
        return 'Wealth Tax'
    elif tax in ['property tax', 'estate tax', 'real estate tax', 'property acquisition tax', 'tax on unbuilt urban land', 'property tax (specifically business rates)', 'property tax returns', 'rental income tax', 'real estate taxation']:
        return 'Property Tax'
    elif tax in ['value added tax', 'value added tax (vat)', 'value-added tax', 'vat (value added tax)', 'state value-added tax', 'tax on transactions recorded in tax invoices', 'value-added tax (vat)', 'value added tax, withholding tax (eg based on the context of businesses and types of taxes mentioned)', 'vat (value added tax), corporate tax, customs duties']:
        return 'Value Added Tax (VAT)'
    elif tax in ['goods and services tax', 'goods and services tax (gst)', 'goods and service tax (gst)', 'goods and service tax', 'service tax', 'service taxes', 'service tax system', 'tax on the circulation of goods and services (icms)', 'icms (tax on operations related to the circulation of goods and on the provision of interstate and intercity transport and communication services)', 'state value-added tax (icms)']:
        return 'Goods and Services Tax (GST)'
    elif tax in ['sales tax', 'tax on sales', 'sales tax, vat']:
        return 'Sales Tax'
    elif tax in ['customs tax', 'tariff', 'customs revenues', 'import tax', 'export tax', 'customs revenue', 'customs-related income tax', 'customs duties', 'customs duties and taxes', 'customs tariff classification tax', 'double tax treaty', 'double taxation', 'foreign buyer taxes', 'customs-related income taxes', 'customs taxation']:
        return 'Trade & Customs Tax'
    elif tax in ['carbon tax', 'carbon tax (ct)', 'environmental tax', 'eco-taxation', 'eco-taxation (specifically on emissions within agricultural supply chains)', 'fuel tax', 'mining tax', 'water resource tax', 'carbon taxes', 'environmental protection tax', 'environmental taxes (fuel, road usage, carbon emissions)', 'water resource fee-to-tax reform', 'minerals resource rent tax (originally resources super profits tax)', 'environmental taxes']:
        return 'Environmental Tax'
    elif tax in ['soda tax', 'sugar tax', 'health insurance', 'taxation on sugar sweetened beverages', 'health insurance copay', 'taxation on sugar sweetened beverages (ssb)', 'soda tax (excise tax on sugar-sweetened beverages)', 'student loan interest deduction (slid)']:
        return 'Health Tax'
    elif tax in ['pigovian tax', 'toll tax', 'vehicle tax', 'motor vehicle tax', 'occupancy tax', 'parking tax', 'tax on specific reckless behaviors', 'tax on unbuilt urban land', 'tax on specific reckless behaviors (e.g., texting while driving)']:
        return 'Other Excise Taxes'
    elif tax in ['progressive income tax, corporate tax', 'corporate tax, sales tax', 'u.s. federal income tax, corporate income tax', 'total tax, federal excise duty, direct tax, sales tax, custom duties', 'direct tax, indirect tax, non-tax revenue']:
        return 'Multiple taxes'
    else:
        return 'Not specified'

# Apply the categorization function
df_merged_results['Exact Tax Categorized'] = df_merged_results['Exact Tax'].apply(categorize_exact_tax)

# Filter for entries categorized as "Other"
other_entries = df_merged_results[df_merged_results['Exact Tax Categorized'] == 'Not specified']
other_entries = other_entries['Exact Tax']

# Calculate the value counts for each category
exact_tax_counts = df_merged_results['Exact Tax Categorized'].value_counts()

# Calculate the percentage of each category
exact_tax_percentages = (exact_tax_counts / exact_tax_counts.sum()) * 100

# Print the counts and percentages
print("\nExact Tax Counts:\n", exact_tax_counts)
print("\nExact Tax Percentages:\n", exact_tax_percentages)




# Machine Learning Analysis:
# Extract the specified columns into a new DataFrame
ml_analysis_df = df_merged_results[['Type of ML Refinement', 'Specific Technique', 'Type of ML']]

# Correct errors from NLP
df_merged_results.loc[df_merged_results['Specific Technique'] == 'Optimization', 'Type of ML Refinement'] = 'Reinforcement Learning'
df_merged_results.loc[df_merged_results['Specific Technique'] == 'Numerical Simulation', 'Type of ML Refinement'] = 'Reinforcement Learning'
df_merged_results.loc[df_merged_results['Specific Technique'] == 'Graph Convolutional Networks', 'Type of ML Refinement'] = 'Semi-Supervised Learning'


type_of_ml = df_merged_results['Type of ML Refinement'].value_counts()
print("\nType of ML:\n", type_of_ml)

# Update 'Type of ML Refinement' column
df_merged_results['Type of ML Refinement'] = df_merged_results['Type of ML Refinement'].replace({
    "Hybrid (Supervised Learning and Unsupervised Learning)": "Supervised Learning, Unsupervised Learning",
    "Supervised Machine Learning": "Supervised Learning"
})
type_of_ml = df_merged_results['Type of ML Refinement'].value_counts()
type_of_ml_percentages = (type_of_ml / type_of_ml.sum()) * 100
print("\nType of ML:\n", type_of_ml)
print("\nType of ML Percentages:\n", type_of_ml_percentages)


# Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(type_of_ml_percentages, labels=type_of_ml.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(type_of_ml)))
plt.title('Distribution of ML Types')
plt.show()


# Analyse specific ML techniques:
specific_technique = df_merged_results['Specific Technique'].value_counts()
print("\nSpecific Technique:\n", specific_technique)

# Function to categorize specific techniques
def categorize_specific_technique(technique):
    technique = technique.lower()
    if 'predictive modeling (classification), data mining (clustering)' in technique or 'classification, clustering' in technique or 'classification (supervised module), clustering (unsupervised module)' in technique:
        return 'Classification & Clustering'
    elif 'classification, regression, clustering' in technique:
        return 'Classification, Regression & Clustering'
    elif 'clustering' in technique or 'topic modeling' in technique:
        return 'Clustering'
    elif 'dimensionality reduction' in technique:
        return 'Dimensionality Reduction'
    elif 'anomaly detection' in technique:
        return 'Anomaly Detection'
    elif 'adversarial networks' in technique:
        return 'Generative Models'
    elif 'classification' in technique or 'sentiment analysis' in technique or 'semantic segmentation' in technique or 'risk score' in technique or 'textual analysis' in technique or 'survival analysis' in technique or 'encoder-decoder' in technique or 'predictive modeling' in technique or 'predictive analysis' in technique:
        return 'Classification'
    elif 'regression' in technique or 'mixture of regressions' in technique or 'prediction' in technique or 'causal machine learning' in technique or 'causal forest' in technique or 'variable selection' in technique or 'multilevel modelling' in technique:
        return 'Regression'
    elif 'time series' in technique or 'forecasting' in technique or 'sequence modeling' in technique or 'break detection' in technique:
        return 'Time Series Forecasting'
    elif 'ranking' in technique:
        return 'Ranking'
    elif 'ensemble methods' in technique:
       return 'Ensemble Methods'
    elif 'graph convolutional networks' in technique:
        return 'Graph Based Methods'
    elif 'model-based reinforcement learning' in technique or 'numerical simulation' in technique:
        return 'Model-Based RL'
    elif 'q-learning' in technique or 'optimization' in technique or 'policy optimization' in technique or 'partially observable markov decision process' in technique or 'pomdp' in technique or 'multi-agent reinforcement learning' in technique or 'deep reinforcement learning' in technique or 'reward shaping' in technique or 'genetic algorithms' in technique:
        return 'Model-Free RL'
    else:
        return 'Not specified'


# Apply the function to the 'Specific Technique' column
df_merged_results['Specific Technique Categorized'] = df_merged_results['Specific Technique'].apply(categorize_specific_technique)

# Count the occurrences of each categorized technique
specific_technique_categorized = df_merged_results['Specific Technique Categorized'].value_counts()
specific_technique_categorized_percentages = (specific_technique_categorized / specific_technique_categorized.sum()) * 100

# Print the counts and percentages of categorized techniques
print("\nSpecific Technique Categorized:\n", specific_technique_categorized)
print("\nSpecific Technique Categorized Percentages:\n", specific_technique_categorized_percentages)


# Group by 'Type of ML Refinement' and get the counts of 'Specific Technique Categorized'
specific_technique_counts = df_merged_results.groupby('Type of ML Refinement')['Specific Technique Categorized'].value_counts().unstack().fillna(0)

print("\nSpecific Technique Categorized for each Type of ML Refinement:\n", specific_technique_counts)

# Deep Learning 
deep_learning = df_merged_results['Deep Learning'].value_counts()
print("\nDeep Learning:\n", deep_learning)

# Normalize 'Deep Learning' column to only "Yes" or "No"
df_merged_results['Deep Learning'] = df_merged_results['Deep Learning'].apply(lambda x: "Yes" if "Yes" in x else "No")
deep_learning = df_merged_results['Deep Learning'].value_counts()
deep_learning_percentages = (deep_learning / deep_learning.sum()) * 100
print("\nDeep Learning:\n", deep_learning)
print("\nDeep Learning Percentages:\n", deep_learning_percentages)


# main_category_type = df_merged_results.groupby('Type of ML Refinement')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())
# main_category_technique = df_merged_results.groupby('Specific Technique Categorized')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())

# print("\nMain 'Category of Research Focus Refinement' for each 'ML type':\n", main_category_type)
# print("\nMain 'Category of Research Focus Refinement' for each 'ML technique':\n", main_category_technique)

# Main variable distinction for each 'Category of Research Focus Refinement'
main_type_category = df_merged_results.groupby('Category of Research Focus Refinement')['Type of ML Refinement'].agg(lambda x: x.value_counts().idxmax())
main_technique_category = df_merged_results.groupby('Category of Research Focus Refinement')['Specific Technique Categorized'].agg(lambda x: x.value_counts().idxmax())

print("\nMain 'ML type' for each 'Category of Research Focus Refinement':\n", main_type_category)
print("\nMain 'ML technique' for each 'Category of Research Focus Refinement':\n", main_technique_category)

# main_type_year = df_merged_results.groupby('Year')['Type of ML Refinement'].agg(lambda x: x.value_counts().idxmax())
# print("\nMain 'ML type' for each year:\n", main_type_year)

# main_category_year = df_merged_results.groupby('Year')['Specific Technique Categorized'].agg(lambda x: x.value_counts().idxmax())
# print("\nMain 'ML type' for each year:\n", main_category_year)

# main_DL_year = df_merged_results.groupby('Year')['Deep Learning'].agg(lambda x: x.value_counts().idxmax())
# print("\nMain 'ML type' for each year:\n", main_DL_year)


# Sanky Diagram
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Prepare data for the Sankey diagram
df_sankey = df_merged_results.copy()

# Exclude "Not specified" category
df_sankey = df_sankey[df_sankey['Specific Technique Categorized'] != 'Not specified']

# Rename "Supervised Learning, Unsupervised Learning" to "Supervised Learning and Unsupervised Learning"
df_sankey['Type of ML Refinement'] = df_sankey['Type of ML Refinement'].replace({
    "Supervised Learning, Unsupervised Learning": "Supervised Learning and Unsupervised Learning"
})

# Adjust counts for "Clustering & Classification" and "Classification, Regression & Clustering"
df_sankey['Specific Technique Categorized'] = df_sankey['Specific Technique Categorized'].replace({
    "Classification & Clustering": "Classification",
    "Classification, Regression & Clustering": "Classification"
})


# Create a DataFrame for additional counts
additional_data = pd.DataFrame([
    {'Specific Technique Categorized': 'Clustering', 'Type of ML Refinement': 'Supervised Learning and Unsupervised Learning', 'Category of Research Focus Refinement': None},
    {'Specific Technique Categorized': 'Clustering', 'Type of ML Refinement': 'Supervised Learning and Unsupervised Learning', 'Category of Research Focus Refinement': None},
    {'Specific Technique Categorized': 'Clustering', 'Type of ML Refinement': 'Supervised Learning and Unsupervised Learning', 'Category of Research Focus Refinement': None},
    {'Specific Technique Categorized': 'Clustering', 'Type of ML Refinement': 'Supervised Learning and Unsupervised Learning', 'Category of Research Focus Refinement': None},
    {'Specific Technique Categorized': 'Regression', 'Type of ML Refinement': 'Supervised Learning and Unsupervised Learning', 'Category of Research Focus Refinement': None}
])

# Concatenate the additional data to the original DataFrame
df_sankey = pd.concat([df_sankey, additional_data], ignore_index=True)

# Define the desired order for the techniques
desired_order = [
    'Classification', 'Regression', 'Time Series Forecasting', 'Ranking', 'Ensemble Methods',
    'Graph Based Methods', 'Clustering', 'Anomaly Detection', 'Dimensionality Reduction',
    'Generative Models', 'Model-Free RL', 'Model-Based RL'
]

# Sort the techniques according to the desired order
df_sankey['Specific Technique Categorized'] = pd.Categorical(df_sankey['Specific Technique Categorized'], categories=desired_order, ordered=True)
df_sankey = df_sankey.sort_values('Specific Technique Categorized')

# Prepare data for the Sankey diagram
techniques = df_sankey['Specific Technique Categorized'].unique()
categories = df_sankey['Category of Research Focus Refinement'].dropna().unique()
ml_types = df_sankey['Type of ML Refinement'].unique()

# Create a mapping from technique/category to index
technique_to_index = {technique: i for i, technique in enumerate(techniques)}
ml_type_to_index = {ml_type: i + len(techniques) for i, ml_type in enumerate(ml_types)}
category_to_index = {category: i + len(techniques) + len(ml_types) for i, category in enumerate(categories)}

# Create source, target, and value lists for the Sankey diagram
source = []
target = []
value = []

# Add links from techniques to ML types
for technique in techniques:
    for ml_type in ml_types:
        count = df_sankey[(df_sankey['Specific Technique Categorized'] == technique) & 
                          (df_sankey['Type of ML Refinement'] == ml_type)].shape[0]
        if count > 0:
            source.append(technique_to_index[technique])
            target.append(ml_type_to_index[ml_type])
            value.append(count)

# Add links from ML types to categories
for ml_type in ml_types:
    for category in categories:
        count = df_sankey[(df_sankey['Type of ML Refinement'] == ml_type) & 
                          (df_sankey['Category of Research Focus Refinement'] == category)].shape[0]
        if count > 0:
            source.append(ml_type_to_index[ml_type])
            target.append(category_to_index[category])
            value.append(count)

# Create labels for the Sankey diagram
labels = list(techniques) + list(ml_types) + list(categories)

# Define colors for the nodes
technique_colors = ['#C6E2FF', '#B9D3EE', '#9FB6CD', '#708090', '#4A708B', '#6C7B8B', '#C6E2FF', '#B9D3EE', '#9FB6CD', '#708090',  '#4A708B','#6C7B8B' ]  # Darkish blue shades for techniques
ml_type_colors = ['#ADE6E6', '#7AC5CD', '#8EE5EE', '#5F9EA0', '#53868B']  # shades of blue for ML types
category_colors = ['#9BCD9B', '#C1FFC1', '#698B69', '#B4EEB4',  '#8FBC8F']  # shades of green for categories
node_colors = technique_colors + ml_type_colors + category_colors


# Define colors for the links, slightly transparent
link_colors = []
for s in source:
    if s < len(techniques):
        color = technique_colors[s]
    elif s < len(techniques) + len(ml_types):
        color = ml_type_colors[s - len(techniques)]
    else:
        color = category_colors[s - len(techniques) - len(ml_types)]
    rgba_color = f'rgba({int(int(color[1:3], 16))}, {int(int(color[3:5], 16))}, {int(int(color[5:7], 16) )}, 0.5)'
    link_colors.append(rgba_color)

# Sankey Diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.2),
        label=labels,
        color=node_colors,
        hoverlabel=dict(font_size=14)  
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
))

# Add descriptions under the left, middle, and right parts
fig.update_layout(
    font=dict(size=14),
    annotations=[
        dict(
            x=0.0,
            y=-0.1,
            text="<b>ML Technique</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=14)
        ),
        dict(
            x=0.5,
            y=-0.1,
            text="<b>ML Type</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=14)
        ),
        dict(
            x=1.0,
            y=-0.1,
            text="<b>Application in Taxation Research</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=14)
        )
    ]
)

fig.show()

# Save the plot as an HTML file and open it in a web browser
pio.write_html(fig, file='sankey_diagram.html', auto_open=True)


# Research focus and ML Type clustering
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Ensure the 'Category of Research Focus Refinement' and 'Type of ML Refinement' columns are not null
df_cluster = df_merged_results.dropna(subset=['Category of Research Focus Refinement', 'Type of ML Refinement'])

# Encode the categorical variables
label_encoder_focus = LabelEncoder()
label_encoder_ml = LabelEncoder()

df_cluster['Category of Research Focus Refinement Encoded'] = label_encoder_focus.fit_transform(df_cluster['Category of Research Focus Refinement'])
df_cluster['Type of ML Refinement Encoded'] = label_encoder_ml.fit_transform(df_cluster['Type of ML Refinement'])

# Prepare the data for clustering
X = df_cluster[['Category of Research Focus Refinement Encoded', 'Type of ML Refinement Encoded']]

# Perform K-Means clustering
num_clusters = 5  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df_cluster['Cluster'] = kmeans.fit_predict(X)

# Define cluster labels
cluster_labels = {
    0: "Tax Administration and Enforcement & Supervised Learning",
    1: "Taxpayer Behavior and Perceptions & Supervised Learning",
    2: "Tax Administration and Enforcement & Unsupervised Learning",
    3: "Fiscal Forecasting & Supervised Learning",
    4: "Tax Policy Design & Reinforcement Learning"
}

# Plot the clusters
plt.figure(figsize=(16, 10))  # Adjust the figure size to make it less wide
scatter = sns.scatterplot(data=df_cluster, x='Category of Research Focus Refinement Encoded', y='Type of ML Refinement Encoded', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Clustering of Research Focus and ML Types')
plt.xlabel('Category of Research Focus')
plt.ylabel('Type of ML')
plt.grid(True)

# Add center nodes and connect to outermost nodes
colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
cluster_centers = kmeans.cluster_centers_
for cluster in range(num_clusters):
    cluster_indices = np.where(df_cluster['Cluster'] == cluster)[0]
    cluster_coords = X.iloc[cluster_indices].values
    cluster_center = cluster_centers[cluster]
    
    # Plot the center node
    plt.scatter(cluster_center[0], cluster_center[1], c=[colors[cluster]], marker='x', s=100)
    
    # Identify the outermost nodes using ConvexHull
    if len(cluster_coords) > 2:  # ConvexHull requires at least 3 points
        try:
            hull = ConvexHull(cluster_coords, qhull_options='QJ')
            for simplex in hull.simplices:
                plt.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1], color=colors[cluster])
                for vertex in simplex:
                    plt.plot([cluster_center[0], cluster_coords[vertex, 0]], [cluster_center[1], cluster_coords[vertex, 1]], color=colors[cluster], linestyle=' ')
            
            # Fill the area of the cluster with a transparent color
            hull_points = cluster_coords[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], color=colors[cluster], alpha=0.2)
        except QhullError as e:
            print(f"QhullError for cluster {cluster}: {e}")

# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(num_clusters)]
labels = [cluster_labels[i] for i in range(num_clusters)]
plt.legend(handles, labels, title="Clusters:", fontsize=12, title_fontsize=12, loc='lower left', bbox_to_anchor=(0.0, 0.0), facecolor='white')
plt.show()

# Function to get the top 2 categories and ML types for each cluster
def get_top_2(df, cluster_col, target_col):
    top_2 = df.groupby(cluster_col)[target_col].value_counts().groupby(level=0).head(2).reset_index(name='Count')
    return top_2


# Get the top 2 research categories for each cluster
top_2_research_categories = get_top_2(df_cluster, 'Cluster', 'Category of Research Focus Refinement')

# Get the top 2 ML types for each cluster
top_2_ml_types = get_top_2(df_cluster, 'Cluster', 'Type of ML Refinement')

print("Top 5 Research Categories for Each Cluster:")
print(top_2_research_categories)

print("\nTop 2 ML Types for Each Cluster:")
print(top_2_ml_types)




# Data type and source:
# Data Source
data_type = df_merged_results['Data Type'].value_counts()
print("\nData Source:\n", data_type)

# Recode 'Data Type' column
df_merged_results['Data Type'] = df_merged_results['Data Type'].replace({
    "Numerical, Time-Series (EG based on the nature of tax data and evaluation metrics)": "Numerical, Time-Series",
    "Textual, Numerical": "Numerical, Textual",
    "Numerical, Tabular Data": "Numerical",
    "Time-Series": "Numerical, Time-Series",
    "Numerical, Textual, Time-Series": "Numerical, Textual",
})
data_type = df_merged_results['Data Type'].value_counts()
print("\nData Type:\n", data_type)

# Data Source
data_source = df_merged_results['Data Source'].value_counts()
print("\nData Source:\n", data_source)

# Function to recode 'Data Source' into specified categories
def recode_data_source(source):
    if any(keyword in source.lower() for keyword in ['tax', 'institute', 'municipal','department', 'government', 'revenue', 'authority', 'administration', 'agency', 'audit', 'secretaria','irs', 'fbr', 'customs']):
        return 'Government and Tax Authority Data'
    elif any(keyword in source.lower() for keyword in ['accounting', 'vat', 'company', 'enterprise', 'companies', 'financial', 'finance', 'corporate', 'business', 'company', 'firm', 'market', 'transaction', 'economic', 'revenue', 'profit', 'statement', 'stock', 'cash registers', 'conference call ']):
        return 'Corporate and Financial Data'
    elif any(keyword in source.lower() for keyword in ['legal', 'law', 'regulatory', 'court', 'case', 'compliance']):
        return 'Legal and Regulatory Data'
    elif any(keyword in source.lower() for keyword in ['social media', 'newspaper', 'twitter', 'facebook', 'linkedin', 'weibo', 'instagram', 'blog', 'forum', 'sentiment']):
        return 'Social Media and Public Sentiment Data'
    elif any(keyword in source.lower() for keyword in ['simulation', 'synthetic', 'simulated', 'agent-based', 'model']):
        return 'Simulation and Synthetic Data'
    elif any(keyword in source.lower() for keyword in ['not specified', 'unspecified']):
        return 'Not specified'
    else:
        return 'Specialized Data Sources'

# Apply the function to the 'Data Source' column
df_merged_results['Data Source Categorized'] = df_merged_results['Data Source'].apply(recode_data_source)

# Get the value counts for the recoded 'Data Source'
data_source_counts = df_merged_results['Data Source Categorized'].value_counts()
total_count = data_source_counts.sum()
data_source_percentages = (data_source_counts / total_count) * 100

# Print the recoded data source counts and percentages
print("\nData Source Categorized:\n", data_source_counts)
print("\nData Source Percentages:\n", data_source_percentages)

# Plot the data
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
ax = sns.barplot(y=data_source_percentages.index, x=data_source_percentages.values, palette="Blues_r")

# Annotate the bars with the percentage values
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 1, p.get_y() + p.get_height() / 2, f'{width:.1f}%', ha='left', va='center')

# Update x-axis label to show percentage
ax.set_xlabel('Percentage')

# Set x-axis ticks to show percentages up to 70%
ax.set_xlim(0, 70)
ax.set_xticks(range(0, 70 + 1, int(70 / 10)))
ax.set_xticklabels([f'{tick}%' for tick in ax.get_xticks()])

#plt.title('Categorized Data Sources')
plt.ylabel('Data Source Categories')
plt.show()

# Most used 'Category of Research Focus Refinement' and 'Type of ML Refinement' for each variable
# category_focus_type = df_merged_results.groupby('Data Type')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())
# type_focus_type = df_merged_results.groupby('Data Type')['Type of ML Refinement'].agg(lambda x: x.value_counts().idxmax())
# category_focus_source = df_merged_results.groupby('Data Source')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())
# type_focus_source = df_merged_results.groupby('Data Source')['Type of ML Refinement'].agg(lambda x: x.value_counts().idxmax())

# print("\nMost used 'Category of Research Focus Refinement' for each 'Data Type':\n", category_focus_type)
# print("\nMost used 'Type of ML Refinement' for each 'Data Type':\n", type_focus_type)
# print("\nMost used 'Category of Research Focus Refinement' for each 'Data Source':\n", category_focus_source)
# print("\nMost used 'Type of ML Refinement' for each 'Data Source':\n", type_focus_source)

# Main variable distinction for each 'Category of Research Focus Refinement' and 'Type of ML Refinement'
main_type_category = df_merged_results.groupby('Category of Research Focus Refinement')['Data Type'].agg(lambda x: x.value_counts().idxmax())
main_source_category = df_merged_results.groupby('Category of Research Focus Refinement')['Data Source Categorized'].agg(lambda x: x.value_counts().idxmax())

main_type_type = df_merged_results.groupby('Type of ML Refinement')['Data Type'].agg(lambda x: x.value_counts().idxmax())
main_type_source = df_merged_results.groupby('Type of ML Refinement')['Data Source Categorized'].agg(lambda x: x.value_counts().idxmax())

print("\nMain 'Data Type' for each 'Category of Research Focus Refinement':\n", main_type_category)
print("\nMain 'Data Source' for each 'Category of Research Focus Refinement':\n", main_source_category)

print("\nMain 'Data Type' for each 'Type of ML Refinement':\n", main_type_type)
print("\nMain 'Data Source' for each 'Type of ML Refinement':\n", main_type_source)


# Geographical focus:import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import geopandas as gpd

# Normalize and categorize geographical focus
def categorize_geographical_focus(focus):
    focus = focus.lower()
    if 'global' in focus:
        return 'Global'
    elif 'multiple' in focus:
        return 'Global'
    elif 'asia' in focus:
        return 'Asia'
    elif 'north america' in focus:
        return 'North America'
    elif 'south america' in focus:
        return 'South America'
    elif 'europe' in focus:
        return 'Europe'
    elif 'africa' in focus:
        return 'Africa'
    elif 'australia' in focus:
        return 'Australia and Oceania'
    elif 'oceania' in focus:
        return 'Australia and Oceania'
    else:
        return 'Not specified'

df_merged_results['Geographical Focus Categorized'] = df_merged_results['Geographical Focus'].apply(categorize_geographical_focus)
geographical_focus_categorized = df_merged_results['Geographical Focus Categorized'].value_counts()

# Calculate the percentage for each continent
total_continent_count = geographical_focus_categorized.sum()
geographical_focus_percentages = (geographical_focus_categorized / total_continent_count) * 100

print("\nGeographical Focus Categorized:\n", geographical_focus_categorized)
print("\nGeographical Focus Percentages:\n", geographical_focus_percentages)

# Function to clean and extract the country from the 'Geographical Focus' column
def extract_country(geo_focus):
    # Remove any specific mentions before the country
    geo_focus = re.sub(r'specifically\s+', '', geo_focus, flags=re.IGNORECASE)
    geo_focus = re.sub(r'\(.*?\)', '', geo_focus)  # Remove content within parentheses
    geo_focus = geo_focus.strip()
    # Extract the country (assuming the format is "continent, country")
    parts = geo_focus.split(',')
    if len(parts) > 1:
        country = parts[1].strip()
        country = country.replace("Republic of Korea", "South Korea")
        country = country.replace("United States of America", "United States")
        country = country.replace("United States", "United States of America")
        country = country.replace("Telangana", "India")
        # Remove entries which do not start with a capitalized letter
        if not country[0].isupper():
            return np.nan
        return country
    return np.nan

# Apply the function to the 'Geographical Focus' column
df_merged_results['Country'] = df_merged_results['Geographical Focus'].apply(extract_country)

# Replace specific entries with NaN
df_merged_results['Country'] = df_merged_results['Country'].replace(['Europe', 'Global', 'Oceania'], np.nan)

# Count the occurrences of each country, ignoring NaN values
country_counts = df_merged_results['Country'].value_counts(dropna=True)

# Calculate the percentage for each country
total_country_count = country_counts.sum()
country_percentages = (country_counts / total_country_count) * 100

# Print the counts and percentages of all countries
print("\nCounts of all countries:")
print(country_counts)
print("\nPercentages of all countries:")
print(country_percentages)

# Get the list of unique countries
unique_countries = df_merged_results['Country'].unique()

# Load the world map shapefile
world = gpd.read_file(r'data/ne_110m_admin_0_countries.shp')  

# Create a column to highlight the countries in the list
world['highlight'] = world['NAME'].apply(lambda x: 1 if x in unique_countries else 0)

# Plot the world map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax, edgecolor='darkgray')
world[world['highlight'] == 1].plot(ax=ax, color='lightblue', edgecolor='black')

# Remove grid and axis labels
ax.grid(False)
ax.set_axis_off()
plt.show()


# Overview of the most used 'Category of Research Focus Refinement' and 'Type of ML Refinement' in each geographical focus
#category_focus = df_merged_results.groupby('Geographical Focus Categorized')['Category of Research Focus Refinement'].agg(lambda x: x.value_counts().idxmax())
#type_focus = df_merged_results.groupby('Geographical Focus Categorized')['Type of ML Refinement'].agg(lambda x: x.value_counts().idxmax())
#print("\nMost used 'Category of Research Focus Refinement' in each geographical focus:\n", category_focus)
#print("\nMost used 'Type of ML Refinement' in each geographical focus:\n", type_focus)

# Main geographical focus for each 'Category of Research Focus Refinement' and 'Type of ML Refinement'
main_geo_focus_category = df_merged_results.groupby('Category of Research Focus Refinement')['Geographical Focus Categorized'].agg(lambda x: x.value_counts().idxmax())
main_geo_focus_type = df_merged_results.groupby('Type of ML Refinement')['Geographical Focus Categorized'].agg(lambda x: x.value_counts().idxmax())
print("\nMain geographical focus for each 'Category of Research Focus Refinement':\n", main_geo_focus_category)
print("\nMain geographical focus for each 'Type of ML Refinement':\n", main_geo_focus_type)


# Analysis of Key Finings
# Convert all float columns to strings
df_merged_results = df_merged_results.applymap(lambda x: str(x) if isinstance(x, float) else x)

# Normalize and categorize key findings
def categorize_key_findings(finding):
    finding = finding.lower()
    if 'specified' in finding or 'mentioned' in finding or 'nan' in finding:
        return 'Not specified'
    else:
        return 'Other'

df_merged_results['Key Findings Categorized'] = df_merged_results['Key Findings'].apply(categorize_key_findings)
key_findings_categorized = df_merged_results['Key Findings Categorized'].value_counts()

# Filter for entries categorized as "Other"
other_key_findings = df_merged_results[df_merged_results['Key Findings Categorized'] == 'Other']
other_key_findings = other_key_findings['Key Findings']

# Calculate the percentage for each category
total_key_findings_count = key_findings_categorized.sum()
key_findings_categorized_percentages = (key_findings_categorized / total_key_findings_count) * 100

print("\nKey Findings Categorized:\n", key_findings_categorized)
print("\nKey Findings Categorized Percentages:\n", key_findings_categorized_percentages)



# Analysis of Challenges and Limitations:
# Convert all float columns to strings
df_merged_results = df_merged_results.applymap(lambda x: str(x) if isinstance(x, float) else x)

# Normalize and categorize challenges/limitations
def categorize_limitations(focus):
    focus = focus.lower()
    if 'specified' in focus or 'mentioned' in focus or 'nan' in focus or 'specific challenges' in focus or 'challenges/limitations:' in focus:
        return 'Not specified'
    else:
        return 'Other'

df_merged_results['Challenges/Limitations Categorized'] = df_merged_results['Challenges/Limitations'].apply(categorize_limitations)
limitations_categorized = df_merged_results['Challenges/Limitations Categorized'].value_counts()

# Filter for entries categorized as "Other"
other_entries = df_merged_results[df_merged_results['Challenges/Limitations Categorized'] == 'Other']
other_entries = other_entries['Challenges/Limitations']

# Calculate the percentage for each category
total_limitations_count = limitations_categorized.sum()
limitations_categorized_percentages = (limitations_categorized / total_limitations_count) * 100

print("\nChallenges/Limitations Categorized:\n", limitations_categorized)
print("\nChallenges/Limitations Percentages:\n", limitations_categorized_percentages)


# Define the output file path
output_file_path = r"data/merged_search_results_v6.xlsx"
df_merged_results.to_excel(output_file_path, index=False)


# Create Bibtex 
import pandas as pd

# Define the file path
file_path = r"data/merged_search_results_v5.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Function to create a unique reference name
def create_reference_name(row):
    authors = row["Author(s)"].split(";")[0].split(",")[0]  # Use the last name of the first author
    year = row["Year"]
    return f"{authors}_{year}"

# Function to create a BibTeX entry for each row
def create_bibtex_entry(row):
    entry_type = "article" if row['Document type'].lower() == "article" else "inproceedings"
    reference_name = create_reference_name(row)
    authors = row["Author(s)"].replace(";", " and")
    title = row["Title"]
    journal = row["Journal"]
    year = row["Year"]
    volume = row["Volume"]
    issue = row["Issue"]
    pages = row["Pages"]
    
    bibtex_entry = f"@{entry_type}{{{reference_name},\n"
    bibtex_entry += f"  author = {{{authors}}},\n"
    bibtex_entry += f"  title = {{{title}}},\n"
    if entry_type == "article":
        bibtex_entry += f"  journal = {{{journal}}},\n"
    else:
        bibtex_entry += f"  booktitle = {{{journal}}},\n"
    bibtex_entry += f"  year = {{{year}}},\n"
    if pd.notna(volume):
        bibtex_entry += f"  volume = {{{volume}}},\n"
    if pd.notna(issue):
        bibtex_entry += f"  number = {{{issue}}},\n"
    if pd.notna(pages):
        bibtex_entry += f"  pages = {{{pages}}},\n"
    bibtex_entry += "}"
    
    return bibtex_entry

# Apply the function to each row and create a new column with BibTeX entries
df['BibTeX'] = df.apply(create_bibtex_entry, axis=1)

# Define the output file path
output_file_path = r"data/merged_search_results_v5_with_bibtex.xlsx"

# Save the DataFrame with BibTeX entries to a new Excel file
df.to_excel(output_file_path, index=False)

print(f"DataFrame with BibTeX entries saved to {output_file_path}")


