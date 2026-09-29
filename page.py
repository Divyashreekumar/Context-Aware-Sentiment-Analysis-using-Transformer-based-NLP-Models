import streamlit as st
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from googleapiclient.discovery import build
import re
from amazon_review import get_product_review
from predict import predict_text
from transformers import PreTrainedTokenizerFast
import pandas as pd

load_dotenv(override=True)
# Modern color palette
SENTIMENT_COLORS = {
    'Positive': '#2ecc71',    # Vibrant Green
    'Negative': '#e74c3c',    # Bright Red
    'Neutral': '#3498db'      # Calm Blue
}

tokenizer = PreTrainedTokenizerFast(tokenizer_file=r'intent classification1\tokenizer.json')

def get_youtube_comments(api_key, video_id, max_comments):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Retrieve video comments
    comments = []
    nextPageToken = None
    total_comments_retrieved = 0

    while True and total_comments_retrieved < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=nextPageToken,
            order='relevance',
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            total_comments_retrieved += 1

            if total_comments_retrieved >= max_comments:
                break

        nextPageToken = response.get('nextPageToken')

        if not nextPageToken or total_comments_retrieved >= max_comments:
            break

    return comments

def extract_video_id(url):
    # Regular expression to extract the video ID from various YouTube URL formats
    video_id_match = re.search(r"(?:v=|v\/|vi\/|videos\/|embed\/|youtu.be\/|watch\?v=|\?v=|&v=|\?id=)([a-zA-Z0-9_-]+)", url)

    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

def process_single_comment(comment):
    """
    Process a single comment and return its sentiment
    """
    # Preprocess comment
    processed_comment = tokenizer.decode(tokenizer.encode(comment, max_length=510, truncation=True))
    
    # Predict sentiment
    result = predict_text([processed_comment])[0]
    
    return result

def process_comments_in_batches(comments, platform, batch_size=50):
    # Initialize progress bar
    progress_bar = st.progress(0)
    
    # Initialize lists to track sentiment counts
    positive_counts = []
    negative_counts = []
    neutral_counts = []
    
    # Create placeholders for charts
    bar_chart_placeholder = st.empty()
    pie_chart_placeholder = st.empty()
    
    # Process comments in batches
    for i in range(0, len(comments), batch_size):
        # Calculate progress
        progress = min((i + batch_size) / len(comments), 1.0)
        progress_bar.progress(progress)
        
        # Get current batch of comments
        batch = comments[i:i+batch_size]
        
        # Preprocess comments
        processed_batch = [tokenizer.decode(tokenizer.encode(comment, max_length=510, truncation=True)) for comment in batch]
        
        # Predict sentiment for the batch
        results = predict_text(processed_batch)
        
        # Calculate sentiment counts for this batch
        batch_positive = sum(1 for r in results if r.label == "Positive")
        batch_negative = sum(1 for r in results if r.label == "Negative")
        batch_neutral = sum(1 for r in results if r.label == "Neutral")
        
        # Append batch counts
        positive_counts.append(batch_positive)
        negative_counts.append(batch_negative)
        neutral_counts.append(batch_neutral)
        
        # Cumulative totals
        total_positive = sum(positive_counts)
        total_negative = sum(negative_counts)
        total_neutral = sum(neutral_counts)
        total_comments = total_positive + total_negative + total_neutral
        
        # Create figure with two subplots
        plt.style.use('bmh')  # Using a built-in Matplotlib style
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f0f0f0')
        fig.patch.set_facecolor('#f0f0f0')
        
        # Color palette
        categories = ["Positive", "Negative", "Neutral"]
        colors = [SENTIMENT_COLORS[cat] for cat in categories]
        
        # Bar Chart
        bar_values = [total_positive, total_negative, total_neutral]
        
        ax1.bar(categories, bar_values, color=colors, edgecolor='white', linewidth=1)
        ax1.set_xlabel('Sentiment Categories', fontweight='bold')
        ax1.set_ylabel('Cumulative Comment Count', fontweight='bold')
        ax1.set_title(f'Sentiment Analysis Progress (Batch {i//batch_size + 1})', fontweight='bold')
        ax1.set_facecolor('#f0f0f0')
        
        # Add value labels on top of each bar with shadow effect
        for j, v in enumerate(bar_values):
            ax1.text(j, v, str(v), ha='center', va='bottom', fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.2'))
        
        # Pie Chart - Percentage
        if total_comments > 0:
            pie_percentages = [
                (total_positive / total_comments) * 100,
                (total_negative / total_comments) * 100,
                (total_neutral / total_comments) * 100
            ]
            
            wedges, texts, autotexts = ax2.pie(
                pie_percentages, 
                labels=categories, 
                colors=colors, 
                autopct='%1.1f%%',
                pctdistance=0.85,
                wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1),
                textprops={'fontweight': 'bold'}
            )
            ax2.set_title('Sentiment Distribution (%)', fontweight='bold')
            ax2.set_facecolor('#f0f0f0')
            
            # Style pie chart text
            plt.setp(autotexts, size=9, weight="bold")
        
        # Adjust layout and display
        if platform == "youtube":
            plt.suptitle('YouTube Comments Sentiment Analysis', fontsize=16, fontweight='bold', backgroundcolor='#f0f0f0')
        else:
            plt.suptitle('Amazon Product Reviews Sentiment Analysis', fontsize=16, fontweight='bold', backgroundcolor='#f0f0f0')
        plt.tight_layout()
        bar_chart_placeholder.pyplot(fig)
        plt.close(fig)
    
    # Final results with styled output
    st.markdown("### 📊 Final Sentiment Analysis Results:")
    if platform == "youtube":
        st.markdown(f"**Total Comments:** {total_comments}")
        st.markdown(f"**🟢 Positive Comments:** {total_positive} "
                    f"({(total_positive/total_comments)*100:.1f}%)")
        st.markdown(f"**🔴 Negative Comments:** {total_negative} "
                    f"({(total_negative/total_comments)*100:.1f}%)")
        st.markdown(f"**🔵 Neutral Comments:** {total_neutral} "
                    f"({(total_neutral/total_comments)*100:.1f}%)")
    else:
        st.markdown(f"**Total Reviews:** {total_comments}")
        st.markdown(f"**🟢 Positive Reviews:** {total_positive} "
                    f"({(total_positive/total_comments)*100:.1f}%)")
        st.markdown(f"**🔴 Negative Reviews:** {total_negative} "
                    f"({(total_negative/total_comments)*100:.1f}%)")
        st.markdown(f"**🔵 Neutral Reviews:** {total_neutral} "
                    f"({(total_neutral/total_comments)*100:.1f}%)")

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊")

    # 👇 INGA indha CSS code paste pannunga
    st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #C7A7FF, #FFD6CC);
    }

    .block-container {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        max-width: 900px;
        margin: 5rem auto;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
    )
#ethu na kudutha comment
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎥 YouTube Comments Analysis", "💬 Single Comment Analysis", "🛒 Amazon Product Analysis"])
    
    # YouTube Comments Analysis Tab
    with tab1:
        # YouTube API key
        
        API_KEY = os.getenv("YOUTUBE_API_KEY")
        
        # Title
        st.title("🎥 YouTube Comment Sentiment Analyzer")
        
        # Prompt for YouTube video URL
        youtube_url = st.text_input("Enter the YouTube video URL")
        
        # Analyze button
        if st.button('Analyze Comments', key="analyze_youtube_btn"):
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            
            if video_id:
                MAX_COMMENTS = 2000  # Maximum comments to retrieve
                
                # Fetch YouTube comments
                with st.spinner('Fetching comments...'):
                    comments = get_youtube_comments(API_KEY, video_id, MAX_COMMENTS)
                
                # Process comments in batches with streaming visualization
                process_comments_in_batches(comments, "youtube")
            else:
                st.error("Invalid YouTube video URL. Please provide a valid URL.")
    
    # Single Comment Analysis Tab
    with tab2:
        st.title("💬 Single Comment Sentiment Analysis")
        
        # Text input for single comment
        single_comment = st.text_area("Enter a comment to analyze its sentiment", height=200)
        
        # Analyze button for single comment
        if st.button('Analyze Comment', key="analyze_single_btn"):
            if single_comment.strip():
                # Process the single comment
                result = process_single_comment(single_comment)
                
                # Display result with color coding
                st.markdown("### Analysis Result:")
                
                # Determine color based on sentiment
                if result.label == "Positive":
                    color = SENTIMENT_COLORS['Positive']
                    emoji = "🟢"
                elif result.label == "Negative":
                    color = SENTIMENT_COLORS['Negative']
                    emoji = "🔴"
                else:
                    color = SENTIMENT_COLORS['Neutral']
                    emoji = "🔵"
                
                # Display sentiment with color and confidence
                st.markdown(f"{emoji} **Sentiment:** <span style='color:{color}'>{result.label}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {result.score:.2%}")
                
                # Display original comment
                st.markdown("### Original Comment:")
                st.write(single_comment)
            else:
                st.warning("Please enter a comment to analyze.")
        # Amazon Product Analysis Tab
    with tab3:
        st.title("🛒 Amazon Product Review Sentiment Analysis")
        
        # Create two columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Amazon product URL input
            amazon_url = st.text_input("Enter Amazon Product URL", 
                                       placeholder="https://www.amazon.com/dp/B0BSHF7LLL")
        
        with col2:
            # Cookie input with password mask
            cookie_input = st.text_input("Enter Amazon Cookies", 
                                        type="password", 
                                        help="Cookies are stored securely and hidden from view")
            
            # Save cookie to session state when entered
            if cookie_input:
                st.session_state.amazon_cookie = cookie_input
                # Success message when cookie is stored
                st.success("Cookie stored securely!")
        
        # Notice about cookie security
        st.info("💡 **Security Notice**: Your Amazon cookie is stored securely in the session and not visible in the UI after entry.")
        if st.button('Analyze Amazon Reviews', key="analyze_amazon_btn"):
            if not amazon_url:
                st.toast("Please enter a valid Amazon product URL", icon="warning")
            if not st.session_state.amazon_cookie:
                st.toast("Please enter your Amazon cookies", icon="warning")
            api_key = os.getenv("unwrangle_api_key",None)
            if not api_key:
                st.toast("API key not found", icon="warning")
            try:
                comments = get_product_review(amazon_url,st.session_state.amazon_cookie,api_key,30)
                print(f"len of comments {len(comments)}")
                process_comments_in_batches(comments, "amazon")
                with st.expander("View Retrieved Reviews"):
                    df = pd.DataFrame(comments, columns=["Comments"])
                    st.write(df)
            except Exception as e:
                st.toast(f"Error", icon="error")


if __name__ == '__main__':
    main()