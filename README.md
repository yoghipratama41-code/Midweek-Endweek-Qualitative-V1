# Mid and End-Week Report Automation

## 📌 Overview
Gathering qualitative information for mid-week and end-week reporting is often a straightforward but highly time-consuming task. It typically involves a tedious back-and-forth process between multiple reports. 

This automation project was designed with one primary goal: **to significantly cut down the processing time** and simplify the workflow.

## ✨ Key Features
- **Streamlined Data Pipeline**: Converts the qualitative information processing into one seamless pipeline. It generates three distinct outputs for each report and automatically compiles a comprehensive final summary.
- **Automated Image Censorship**: Automatically redacts sensitive information within images. This replaces the manual censorship flow, saving a substantial amount of time and effort.

## ⚠️ Important Notes & Limitations
- **Human-in-the-Loop**: Please note that this version is not fully autonomous. It still requires human input for the initial gathering of qualitative information, as well as a final review/check of the outputs.
- **Usage Quotas**: This project is built to be **completely free**. It does not rely on any paid tools or premium APIs. As a result, it is subject to the standard daily usage quotas and rate limits of the underlying free services used to run it.

## 🚀 Getting Started (Usage Steps)
1. **Compile and Organize Images**: Gather the new pictures for both the names and the comments. 
   * *Crucial Note*: The comment image that refers to a main post picture **must have the exact same filename** as that main post picture so the system can match them.
   * *Recommendation*: Separate these images into two distinct folders (e.g., one for main posts and one for comments) to keep the pipeline organized and prevent confusion.
2. **Update the Directory**: Simply place/update these new pictures in your designated working folders, and the automation is ready to process them.

## 🛠️ Tech Stack
- **Python**: The core language used to build the data pipeline and automate the censorship processes.
- **Gemini LLM**: Leveraged to process, synthesize, and summarize the qualitative information efficiently without relying on paid APIs.
