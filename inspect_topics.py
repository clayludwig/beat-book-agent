import json
from collections import Counter

data = json.load(open('source-stories/chicago_public_media_with_topics.json'))

print("=== ALL STORIES WITH TOPICS ===\n")
for s in data:
    topics = s.get('topics', ['?'])
    print(f"{str(topics):<50} {s['title'][:60]}")

print("\n=== BROAD TOPIC DISTRIBUTION ===\n")
broad = Counter(s['topics'][0] for s in data if s.get('topics'))
for label, count in broad.most_common():
    print(f"  {count:3d}  {label}")

print("\n=== SPECIFIC TOPIC DISTRIBUTION ===\n")
specific = Counter(s['topics'][-1] for s in data if s.get('topics'))
for label, count in specific.most_common():
    print(f"  {count:3d}  {label}")
