# Pahalgam Modi Twitter Analysis Workflow Diagram

Below is a detailed Mermaid flowchart diagram representing the workflow in `analyze_pahalgam_modi_with_agents.py`. This diagram visualizes how the multi-agent system processes and analyzes Twitter data related to "Pahalgam Modi".

## Workflow Diagram

```mermaid
graph TD
    %% Main workflow nodes
    Start(["Start Analysis"]) --> Init["Initialize MultiAgentTwitterAnalysis<br/>with 'pahalgam Modi' query"]
    Init --> RunWorkflow["Run Multi-Agent Workflow"]
    RunWorkflow --> SaveResults["Save Results to JSON File"]
    SaveResults --> CreateDF["Extract Tweet Data<br/>to DataFrame"]
    CreateDF --> SaveCSV["Save Data to CSV"]
    SaveCSV --> PrintSummary["Print Analysis Summary"]
    PrintSummary --> End(["End Analysis"])

    %% Multi-agent workflow subgraph
    subgraph MultiAgentWorkflow["Multi-Agent Twitter Analysis Workflow"]
        Step1["Step 1: PromptEnhancer Agent<br/>Refine Query"] --> Step2
        Step2["Step 2: Twitter Scraping<br/>Collect Tweets"] --> Step3
        Step3["Step 3: XInterface Agent<br/>Find Keywords & Trends"] --> Step4
        Step4["Step 4: Screening Agent<br/>Filter Content"] --> Step5
        Step5["Step 5: Save to MongoDB<br/>Store All Tweets"] --> Step6
        Step6["Step 6: DetailedAnalysis Agent<br/>Analyze Sentiment & Topics"]
    end

    %% Connect main flow to subgraph
    RunWorkflow -.-> MultiAgentWorkflow
    MultiAgentWorkflow -.-> SaveResults

    %% Data stores
    MongoDB[(MongoDB<br/>ai_crafted_tweets)] --- Step5
    ResultsJSON[("JSON File<br/>pahalgam_modi_results_*.json")] --- SaveResults
    ResultsCSV[("CSV File<br/>pahalgam_modi_tweets_*.csv")] --- SaveCSV

    %% Agent details in notes
    Step1 -.-> PE["PromptEnhancer:<br/>1. Identifies keywords, hashtags<br/>2. Creates enhanced search query<br/>3. Returns search strategy"]
    Step3 -.-> XI["XInterface:<br/>1. Analyzes tweet content<br/>2. Identifies influential accounts<br/>3. Finds trending hashtags<br/>4. Highlights relevant tweets"]
    Step4 -.-> SA["Screening:<br/>1. Assesses relevance to query<br/>2. Prioritizes content<br/>3. Identifies credibility issues<br/>4. Makes recommendations"]
    Step6 -.-> DA["DetailedAnalysis:<br/>1. Performs sentiment analysis<br/>2. Identifies key story elements<br/>3. Evaluates credibility<br/>4. Provides editorial insights"]
    
    %% Parameters and configuration
    Config["Configuration:<br/>API_KEY = RapidAPI Key<br/>max_tweets = 200<br/>save_all_tweets = True<br/>agent_logging_enabled = False"] --> Init

    %% Styling
    classDef agent fill:#f9d5e5,stroke:#333,stroke-width:1px;
    classDef process fill:#eeeeee,stroke:#333,stroke-width:1px;
    classDef data fill:#d5f5e3,stroke:#333,stroke-width:1px;
    classDef start fill:#d5e8f9,stroke:#333,stroke-width:1px;
    classDef config fill:#fcf3cf,stroke:#333,stroke-width:1px;
    
    class PE,XI,SA,DA agent;
    class Step1,Step2,Step3,Step4,Step5,Step6,Init,RunWorkflow,SaveResults,CreateDF,SaveCSV,PrintSummary process;
    class MongoDB,ResultsJSON,ResultsCSV data;
    class Start,End start;
    class Config config;
```

## How to Use This Diagram

1. Copy the Mermaid code above into a Markdown file or a Mermaid Live Editor
2. View the rendered diagram to understand the workflow
3. Use this as a reference when working with the `analyze_pahalgam_modi_with_agents.py` script

## Key Components Explained

### Main Workflow
- **Start Analysis**: Entry point of the script
- **Initialize MultiAgentTwitterAnalysis**: Creates the workflow object with the "pahalgam Modi" query
- **Run Multi-Agent Workflow**: Executes the 6-step agent workflow
- **Save Results to JSON**: Stores complete analysis results in a JSON file
- **Extract Tweet Data**: Processes tweets into a pandas DataFrame
- **Save Data to CSV**: Exports tweet data to a CSV file
- **Print Analysis Summary**: Outputs a summary of findings to console

### Multi-Agent Workflow
1. **PromptEnhancer Agent**: Refines the user query to identify relevant keywords, hashtags, and accounts
2. **Twitter Scraping**: Uses RapidAPI to collect tweets based on the enhanced query
3. **XInterface Agent**: Analyzes tweets to find trends, influential accounts, and top keywords
4. **Screening Agent**: Evaluates content relevance and credibility, prioritizes important tweets
5. **Save to MongoDB**: Stores all tweets in the MongoDB database
6. **DetailedAnalysis Agent**: Performs sentiment analysis and provides in-depth insights on the content

### Data Stores
- **MongoDB**: Stores the tweet data in the "ai_crafted_tweets" collection
- **JSON Results File**: Contains the complete analysis results
- **CSV File**: Contains the formatted tweet data for easy analysis

### Configuration Parameters
- **API_KEY**: RapidAPI key for Twitter scraping
- **max_tweets**: Maximum number of tweets to collect (set to 200)
- **save_all_tweets**: Flag to save all tweets regardless of screening results (set to True)
- **agent_logging_enabled**: Flag to enable/disable agent logging to MongoDB (set to False to avoid database errors) 