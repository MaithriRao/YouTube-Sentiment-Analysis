// popup.js

document.addEventListener("DOMContentLoaded", async () => {
    // 
    const outputDiv = document.getElementById("output");
    const API_URL = 'http://localhost:5000'; // 

    // --- Key Management Elements ---
    const apiKeySection = document.getElementById('api-key-section'); // Assumes you add this div to popup.html
    const apiKeyInput = document.getElementById('youtube-key'); // Assumes you add this input to popup.html
    const saveButton = document.getElementById('save-key-button'); // Assumes you add this button to popup.html
    
// --- Event Listener for Saving the Key ---
    if (saveButton) {
        saveButton.addEventListener('click', () => {
            const key = apiKeyInput.value.trim();
            if (key) {
                chrome.storage.local.set({ 'youtube_api_key': key }, () => {
                // Check for successful storage save
                if (chrome.runtime.lastError) {
                    console.error("Storage Error:", chrome.runtime.lastError);
                    outputDiv.innerHTML = `<p class='error'>Error saving key: ${chrome.runtime.lastError.message}</p>`;
                    return;
                }
                console.log('YouTube API Key saved successfully.');
                
                // After saving, attempt to run the core logic
                runCoreAnalysisLogic(API_URL);
            });
        } else {
            outputDiv.innerHTML = "<p class='error'>Please enter a valid YouTube API Key.</p>";
        }
        });
    }

    // --- Initial Entry Point ---
    // Instead of running the logic directly, we call a function to check for the key
    runCoreAnalysisLogic(API_URL);
});

// ----------------------------------------------------------------------
// CORE LOGIC CHECKER
// ----------------------------------------------------------------------

async function runCoreAnalysisLogic(API_URL) {
    const outputDiv = document.getElementById("output");
    const apiKeySection = document.getElementById('api-key-section');
    const mainContent = document.getElementById('main-content'); // Assume analysis content is wrapped in a 'main-content' div
    
    // Retrieve the key from storage
    chrome.storage.local.get('youtube_api_key', async (data) => {
        const apiKey = data.youtube_api_key;
        
        if (!apiKey) {
            // Key not found: Show input section, hide results, and stop.
            if (apiKeySection) apiKeySection.style.display = 'block';
            if (mainContent) mainContent.style.display = 'none';
            outputDiv.innerHTML = "<p class='warning'>Please enter your YouTube API Key to begin analysis.</p>";
            return;
        }

        // Key found: Hide input section, show results placeholder.
        if (apiKeySection) apiKeySection.style.display = 'none';
        if (mainContent) mainContent.style.display = 'block';
        outputDiv.innerHTML = "<p>Key loaded. Checking URL...</p>";

        // --- Core Chrome Tab/URL Logic (Modified to pass the retrieved apiKey) ---
        chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
            const url = tabs[0].url;
            const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
            const match = url.match(youtubeRegex);

            if (match && match[1]) {
                const videoId = match[1];
                outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;

                //  Pass the apiKey to fetchComments
                const comments = await fetchComments(videoId, apiKey); 
                if (comments.length === 0) {
                    outputDiv.innerHTML += "<p>No comments found for this video.</p>";
                    return;
                }

                outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;

                const predictions = await getSentimentPredictions(comments, API_URL);

                if (predictions) {
                    // Process the predictions to get sentiment counts and sentiment data
                    const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
                    const sentimentData = []; // For trend graph
                    const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
                    predictions.forEach((item, index) => {
                        sentimentCounts[item.sentiment]++;
                        sentimentData.push({
                            timestamp: item.timestamp,
                            sentiment: parseInt(item.sentiment)
                        });
                    });

                    // Compute metrics
                    const totalComments = comments.length;
                    const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
                    const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
                    const avgWordLength = (totalWords / totalComments).toFixed(2);
                    const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);

                    // Normalize the average sentiment score to a scale of 0 to 10
                    const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

                    // Add the Comment Analysis Summary section
                    outputDiv.innerHTML += `
                      <div class="section">
                        <div class="section-title">Comment Analysis Summary</div>
                        <div class="metrics-container">
                          <div class="metric">
                            <div class="metric-title">Total Comments</div>
                            <div class="metric-value">${totalComments}</div>
                          </div>
                          <div class="metric">
                            <div class="metric-title">Unique Commenters</div>
                            <div class="metric-value">${uniqueCommenters}</div>
                          </div>
                          <div class="metric">
                            <div class="metric-title">Avg Comment Length</div>
                            <div class="metric-value">${avgWordLength} words</div>
                          </div>
                          <div class="metric">
                            <div class="metric-title">Avg Sentiment Score</div>
                            <div class="metric-value">${normalizedSentimentScore}/10</div>
                          </div>
                        </div>
                      </div>
                    `;

                    // Add the Sentiment Analysis Results section with a placeholder for the chart
                    outputDiv.innerHTML += `
                      <div class="section">
                        <div class="section-title">Sentiment Analysis Results</div>
                        <p>See the pie chart below for sentiment distribution.</p>
                        <div id="chart-container"></div>
                      </div>`;

                    // Fetch and display the pie chart inside the chart-container div
                    await fetchAndDisplayChart(sentimentCounts, API_URL);

                    // Add the Sentiment Trend Graph section
                    outputDiv.innerHTML += `
                      <div class="section">
                        <div class="section-title">Sentiment Trend Over Time</div>
                        <div id="trend-graph-container"></div>
                      </div>`;

                    // Fetch and display the sentiment trend graph
                    await fetchAndDisplayTrendGraph(sentimentData, API_URL);

                    // Add the Word Cloud section
                    outputDiv.innerHTML += `
                      <div class="section">
                        <div class="section-title">Comment Wordcloud</div>
                        <div id="wordcloud-container"></div>
                      </div>`;

                    // Fetch and display the word cloud inside the wordcloud-container div
                    await fetchAndDisplayWordCloud(comments.map(comment => comment.text), API_URL);

                    // Add the top comments section
                    outputDiv.innerHTML += `
                      <div class="section">
                        <div class="section-title">Top 25 Comments with Sentiments</div>
                        <ul class="comment-list">
                          ${predictions.slice(0, 25).map((item, index) => `
                            <li class="comment-item">
                              <span>${index + 1}. ${item.comment}</span><br>
                              <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
                            </li>`).join('')}
                        </ul>
                      </div>`;
                }
                
                // ... (END OF ANALYSIS LOGIC) ...

            } else {
                outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
            }
        });
    });
}

// ----------------------------------------------------------------------
// FETCH FUNCTIONS
// ----------------------------------------------------------------------

//  fetchComments now accepts API_KEY
async function fetchComments(videoId, API_KEY) {
    let comments = [];
    let pageToken = "";
    const outputDiv = document.getElementById("output");
    try {
        while (comments.length < 500) {
            // Use the passed API_KEY
            const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
            const data = await response.json();
            
            // Handle API errors (e.g., key invalid, quota exceeded)
            if (data.error) {
                console.error("YouTube API Error:", data.error);
                outputDiv.innerHTML += `<p class='error'>YouTube API Error: ${data.error.message}. Please check your key or quota.</p>`;
                return []; // Return empty array on fatal error
            }

            if (data.items) {
                data.items.forEach(item => {
                    const commentText = item.snippet.topLevelComment.snippet.textOriginal;
                    const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
                    const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
                    comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
                });
            }
            pageToken = data.nextPageToken;
            if (!pageToken) break;
        }
    } catch (error) {
        console.error("Error fetching comments:", error);
        outputDiv.innerHTML += "<p class='error'>Error fetching comments due to network issue.</p>";
    }
    return comments;
}

// Added API_URL to function signature for clarity, though it's accessible globally
async function getSentimentPredictions(comments, API_URL) {
    const outputDiv = document.getElementById("output");
    try {
        const response = await fetch(`${API_URL}/predict_with_timestamps`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ comments })
        });
        const result = await response.json();
        if (response.ok) {
            return result; 
        } else {
            throw new Error(result.error || 'Error fetching predictions');
        }
    } catch (error) {
        console.error("Error fetching predictions:", error);
        outputDiv.innerHTML += "<p class='error'>Error fetching sentiment predictions.</p>";
        return null;
    }
}

//  Added API_URL to function signature for clarity
async function fetchAndDisplayChart(sentimentCounts, API_URL) {
    const outputDiv = document.getElementById("output");
    try {
        const response = await fetch(`${API_URL}/generate_chart`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sentiment_counts: sentimentCounts })
        });
        if (!response.ok) {
            throw new Error('Failed to fetch chart image');
        }
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        const chartContainer = document.getElementById('chart-container');
        chartContainer.appendChild(img);
    } catch (error) {
        console.error("Error fetching chart image:", error);
        outputDiv.innerHTML += "<p class='error'>Error fetching chart image.</p>";
    }
}

// Added API_URL to function signature for clarity
async function fetchAndDisplayWordCloud(comments, API_URL) {
    const outputDiv = document.getElementById("output");
    try {
        const response = await fetch(`${API_URL}/generate_wordcloud`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ comments })
        });
        //
        if (!response.ok) {
            throw new Error('Failed to fetch word cloud image');
        }
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        const wordcloudContainer = document.getElementById('wordcloud-container');
        wordcloudContainer.appendChild(img);
    } catch (error) {
        console.error("Error fetching word cloud image:", error);
        outputDiv.innerHTML += "<p class='error'>Error fetching word cloud image.</p>";
    }
}

// Added API_URL to function signature for clarity
async function fetchAndDisplayTrendGraph(sentimentData, API_URL) {
    const outputDiv = document.getElementById("output");
    try {
        const response = await fetch(`${API_URL}/generate_trend_graph`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sentiment_data: sentimentData })
        });
        // 
        if (!response.ok) {
            throw new Error('Failed to fetch trend graph image');
        }
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        img.style.width = '100%';
        img.style.marginTop = '20px';
        const trendGraphContainer = document.getElementById('trend-graph-container');
        trendGraphContainer.appendChild(img);
    } catch (error) {
        console.error("Error fetching trend graph image:", error);
        outputDiv.innerHTML += "<p class='error'>Error fetching trend graph image.</p>";
    }
}