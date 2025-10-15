"""
@author: kramer

Next, I use Natural Language Processing to verify:
- Category of Research Focus
- Type of Machine Learning used

"""

import subprocess
import json
import pandas as pd
import os

df_merged_results_path = "data/merged_search_results_v3.xlsx"

df_merged_results = pd.read_excel(df_merged_results_path)

# Create a prompt for each column
def create_prompt(row):
    title = row['Title']
    abstract = row['Abstract']
    keywords = row['Keywords']
    prompt = f"Based on the title: '{title}', abstract: '{abstract}', keywords: '{keywords}', fill in the following information:\n"
    prompt += "1.a. Category of Research Focus (Tax Policy Design, Fiscal Forecasting, Tax Administration and Enforcement, Taxpayer Behavior and Perceptions, International Taxation)\n"
    prompt += "1.b. Subcategory of Research Focus (e.g. Tax fraud detection, ...)\n"
    prompt += "2.a. Type of Machine Learning used (Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, Reinforcement Learning)\n"
    prompt += "2.b. Specific ML technique used (e.g. Classification, Clustering, ...).\n"
    prompt += "2.c. Was Deep Learning used? (Yes/No)\n"
    prompt += "Work conscientiously and answer concisely (keywords, no sentences). Start answer with the respective number and letter. If information is not directly given make educated guesses and indicate it by (EG) at the end."
    return prompt

# Apply the function to each row
df_merged_results['Prompt_2'] = df_merged_results.apply(create_prompt, axis=1)

# Save the DataFrame with prompts to an Excel file
df_merged_results.to_excel("data/merged_search_results_v4.xlsx", index=False)

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
responses_df = pd.DataFrame(columns=['Index', 'Category of Research Focus Refinement', 'Subcategory of Research Focus', 'Type of ML Refinement', 'Specific Technique', 'Deep Learning'])

# Apply the function to each prompt and record the results
for index, row in df_merged_results.iterrows():
    prompt = row['Prompt_2']
    response = get_chatgpt_response(prompt)
    response_row = {'Index': index}
    
    tax_focus  = "NA"
    tax_subfocus = "NA"
    type_of_ml = "NA"
    specific_technique = "NA"
    deep_learning = "NA"
    
    if "1.a." in response:
        tax_focus = response.split("1.a.")[1].split("\n")[0].strip()
    if "1.b." in response:  
        tax_subfocus = response.split("1.b.")[1].split("\n")[0].strip()
    if "2.a." in response:
        type_of_ml = response.split("2.a.")[1].split("\n")[0].strip()
    if "2.b." in response:
        specific_technique = response.split("2.b.")[1].split("\n")[0].strip()
    if "2.c." in response:
        deep_learning = response.split("2.c.")[1].split("\n")[0].strip()
    
    response_row['Category of Research Focus Refinement'] = tax_focus 
    response_row['Subcategory of Research Focus'] = tax_subfocus
    response_row['Type of ML Refinement'] = type_of_ml
    response_row['Specific Technique'] = specific_technique
    response_row['Deep Learning'] = deep_learning
    
    responses_df = pd.concat([responses_df, pd.DataFrame([response_row])], ignore_index=True)

# Merge the responses with the original DataFrame
df_merged_results = pd.merge(df_merged_results, responses_df, left_index=True, right_on='Index')


df_merged_results = df_merged_results.drop(columns=['Index'])

# Save the merged DataFrame to an Excel file
df_merged_results.to_excel("data/merged_search_results_v4.xlsx", index=False)
