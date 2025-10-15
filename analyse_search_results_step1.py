"""
@author: kramer

I use Natural Language Processing to exclude results that do not focus on taxation research and the use of machine learning.
The decision is based on information extracted from the title, abstract, and keywords of each publication.

"""

import subprocess
import json
import pandas as pd
import os

df_merged_results_path = "data/merged_search_results_v1.xlsx"

df_merged_results = pd.read_excel(df_merged_results_path)

# Create a prompt for each row
def create_prompt(row):
    title = row['Title']
    abstract = row['Abstract']
    keywords = row['Keywords']
    prompt = f"Based on the title: '{title}', abstract: '{abstract}', keywords: '{keywords}', answer the following questions:\n"
    prompt += "1.a. Does the research focus on taxation or tax-related topics? (Yes/No)\n"
    prompt += "1.b. If Yes for 1.a, specify the category of research focus (e.g., Tax fraud).\n"
    prompt += "2.a. Does the research use Machine Learning as a research method? (Yes/No)\n"
    prompt += "2.b. If No for 2.a, does the research discuss the use of Machine Learning in tax-related practices? (Yes/No)\n"
    prompt += "2.c. If Yes for 2.a, specify the type of Machine Learning used (e.g., Supervised Learning).\n"
    prompt += "Work conscientiously and answer concisely (no whole sentences). Start answer with the respective number and letter."
    return prompt

# Apply the function to each row
df_merged_results['Prompt'] = df_merged_results.apply(create_prompt, axis=1)

# Save the DataFrame with prompts to an Excel file
df_merged_results.to_excel("data/merged_search_results_v2.xlsx", index=False)

# Function to call OpenAI API using curl and get the response
def get_chatgpt_response(prompt):
    api_key = 'API_KEY' # Replace with your OpenAI API key
    model = "gpt-4-turbo"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 50
    }
    response = subprocess.run(
        ["curl", "https://api.openai.com/v1/chat/completions", "-H", f"Content-Type: {headers['Content-Type']}", "-H", f"Authorization: {headers['Authorization']}", "-d", json.dumps(data)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    response_json = json.loads(response.stdout)
    if 'choices' in response_json:
        return response_json['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response_json}")
        return ""

# Initialize a DataFrame to store the responses
responses_df = pd.DataFrame(columns=['Index', 'Focus on Taxation', 'Category of Research Focus', 'Use of ML', 'Discussion of ML', 'Type of ML'])

# Apply the function to each prompt and record the results
for index, row in df_merged_results.iterrows():
    prompt = row['Prompt']
    response = get_chatgpt_response(prompt)
    response_row = {'Index': index}
    
    tax_focus = "No"
    category_of_research_focus = "NA"
    ml_use = "No"
    ml_discussion = "NA"
    type_of_ml = "NA"
    
    if "1.a. Yes" in response:
        tax_focus = "Yes"
        if "1.b." in response:
            category_of_research_focus = response.split("1.b.")[1].split("\n")[0].strip()
    if "2.a. No" in response:
        ml_use = "No"
        if "2.b." in response:
            ml_discussion = response.split("2.b.")[1].split("\n")[0].strip()
    if "2.a. Yes" in response:
        ml_use = "Yes"
        if "2.c." in response:
            type_of_ml = response.split("2.c.")[1].split("\n")[0].strip()
    
    response_row['Focus on Taxation'] = tax_focus
    response_row['Category of Research Focus'] = category_of_research_focus
    response_row['Use of ML'] = ml_use
    response_row['Discussion of ML'] = ml_discussion
    response_row['Type of ML'] = type_of_ml
    
    responses_df = pd.concat([responses_df, pd.DataFrame([response_row])], ignore_index=True)

# Merge the responses with the original DataFrame
df_merged_results = pd.merge(df_merged_results, responses_df, left_index=True, right_on='Index')

df_merged_results = df_merged_results.drop(columns=['Index'])

# Save the merged DataFrame to an Excel file
df_merged_results.to_excel("data/merged_search_results_v2.xlsx", index=False)

# Based on this reult, publications that do not have a focus on tax or taxation and Machine Leanring are manually deleted from the file. The new file is saved as merged_search_results_v3 