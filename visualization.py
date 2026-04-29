import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# BAR CHART - Top Items
# -------------------------------
def plot_top_items(df):
    item_counts = df.sum().sort_values(ascending=False)

    plt.figure()
    item_counts.head(10).plot(kind='bar')
    plt.title("Top 10 Purchased Items")
    plt.xlabel("Items")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------
# PIE CHART - Distribution
# -------------------------------
def plot_item_distribution(df):
    item_counts = df.sum().sort_values(ascending=False)

    plt.figure()
    item_counts.head(5).plot(kind='pie', autopct='%1.1f%%')
    plt.title("Top 5 Item Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# -------------------------------
# SCATTER - Support vs Confidence
# -------------------------------
def plot_support_confidence(rules):
    if rules.empty:
        print("⚠️ No rules for scatter plot")
        return

    plt.figure()
    plt.scatter(rules['support'], rules['confidence'])
    plt.title("Support vs Confidence")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()

# -------------------------------
# HISTOGRAM - Lift Distribution
# -------------------------------
def plot_lift(rules):
    if rules.empty:
        print("⚠️ No rules for histogram")
        return

    plt.figure()
    rules['lift'].plot(kind='hist', bins=10)
    plt.title("Lift Distribution")
    plt.xlabel("Lift")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# -------------------------------
# HEATMAP - Association Rules
# -------------------------------
def plot_heatmap(rules):
    if rules.empty:
        print("⚠️ No rules to display heatmap")
        return

    # Take top 10 rules
    top_rules = rules.sort_values(by='lift', ascending=False).head(10)

    # Convert frozenset → string
    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Create pivot table
    pivot = top_rules.pivot(index='antecedents',
                           columns='consequents',
                           values='lift')

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2f")
    plt.title("Association Rules Heatmap")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()