from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Get sample data
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

# Fit BERTopic
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs[:1000])  # Using subset for speed

# Get topic info
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# Multiple visualizations
fig1 = topic_model.visualize_topics()
fig1.show()

fig2 = topic_model.visualize_barchart(top_k_topics=8)
fig2.show()

fig3 = topic_model.visualize_heatmap()
fig3.show()

fig4 = topic_model.visualize_hierarchy()
fig4.show()

# Documents visualization (requires 2D embeddings)
fig5 = topic_model.visualize_documents(docs[:1000])
fig5.show()
