"""
@author: kramer

I further use Natural Language Processing to extract information on:
1. Type of tax
2. Data type and dataset source
3. Geographical focus
4. Key findings
5. Challenges/Limitations
from the title, abstract, and keywords of each publication.
"""

import subprocess
import json
import pandas as pd
import os

df_merged_results_path = "data/merged_search_results_v4.xlsx"

df_merged_results = pd.read_excel(df_merged_results_path)

# Create a prompt for each column
def create_prompt(row):
    title = row['Title']
    abstract = row['Abstract']
    keywords = row['Keywords']
    prompt = f"Based on the title: '{title}', abstract: '{abstract}', keywords: '{keywords}', fill in the following information:\n"
    prompt += "1.a. Type of tax (Tax on individual income, Tax on individual wealth, Tax on consumption, Tax on business income, Tax on trade)\n"
    prompt += "1.b. Specify exact tax\n"
    prompt += "2.a. Data Type (Textual, Numerical, Time-Series)\n"
    prompt += "2.b. Data Source\n"
    prompt += "3. Geographical focus (Continent and if possible country)\n"
    prompt += "4. Key findings\n"
    prompt += "5. Challenges/Limitations\n"
    prompt += "Work conscientiously and answer concisely (keywords, no sentences). Start answer with the respective number and letter. If information is not directly given make educated guesses and indicate it by (EG) at the end."
    return prompt

# Apply the function to each row
df_merged_results['Prompt_3'] = df_merged_results.apply(create_prompt, axis=1)

# Save the DataFrame with prompts to an Excel file
df_merged_results.to_excel("data/merged_search_results_v5.xlsx", index=False)

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
        "temperature": 0.25,
        "max_tokens": 200
    }
    response = subprocess.run(
        ["curl", "https://api.openai.com/v1/chat/completions", "-H", f"Content-Type: {headers['Content-Type']}", "-H", f"Authorization: {headers['Authorization']}", "-d", json.dumps(data)],
        capture_output=True,
        text=True
    )
    response_json = json.loads(response.stdout)
    if 'choices' in response_json:
        return response_json['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response_json}")
        return ""

# Initialize a DataFrame to store the responses
responses_df = pd.DataFrame(columns=[
    'Index', 'Type of Tax', 'Exact Tax', 
    'Data Type', 'Data Source', 'Geographical Focus', 
    'Key Findings', 'Challenges/Limitations'
])

# Apply the function to each prompt and record the results
for index, row in df_merged_results.iterrows():
    prompt = row['Prompt_3']
    response = get_chatgpt_response(prompt)
    response_row = {'Index': index}
    
    type_of_tax = "NA"
    exact_tax = "NA"
    data_structure = "NA"
    data_type = "NA"
    data_source = "NA"
    geographical_focus = "NA"
    key_findings = "NA"
    challenges_limitations = "NA"
    
    if "1.a." in response:
        type_of_tax = response.split("1.a.")[1].split("\n")[0].strip()
    if "1.b." in response:
        exact_tax = response.split("1.b.")[1].split("\n")[0].strip()
    if "2.a." in response:
        data_type = response.split("2.a.")[1].split("\n")[0].strip()
    if "2.b." in response:
        data_source = response.split("2.b.")[1].split("\n")[0].strip()
    if "3." in response:
        geographical_focus = response.split("3.")[1].split("\n")[0].strip()
    if "4." in response:
        key_findings = response.split("4.")[1].split("\n")[0].strip()
    if "5." in response:
        challenges_limitations = response.split("5.")[1].split("\n")[0].strip()
    
    response_row['Type of Tax'] = type_of_tax
    response_row['Exact Tax'] = exact_tax
    response_row['Data Type'] = data_type
    response_row['Data Source'] = data_source
    response_row['Geographical Focus'] = geographical_focus
    response_row['Key Findings'] = key_findings
    response_row['Challenges/Limitations'] = challenges_limitations
    
    responses_df = pd.concat([responses_df, pd.DataFrame([response_row])], ignore_index=True)

# Merge the responses with the original DataFrame
df_merged_results = pd.merge(df_merged_results, responses_df, left_index=True, right_on='Index')


df_merged_results = df_merged_results.drop(columns=['Index'])

# Save the merged DataFrame to an Excel file
df_merged_results.to_excel("data/merged_search_results_v5.xlsx", index=False)