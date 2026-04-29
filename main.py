from preprocessing import load_data, create_transactions
from analysis import encode_transactions, get_frequent_items, get_rules
from visualization import *

print("🔄 Loading dataset...")
df = load_data()

print("🔄 Creating transactions...")
transactions = create_transactions(df)

print("🔄 Encoding transactions...")
encoded_df = encode_transactions(transactions)

print("🔄 Finding frequent itemsets...")
frequent_items = get_frequent_items(encoded_df)

print("🔄 Generating association rules...")
rules = get_rules(frequent_items)

# -------------------------------
# SHOW SAMPLE RULES
# -------------------------------
print("\n📊 Sample Association Rules:\n")
if not rules.empty:
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
else:
    print("No rules generated. Try lowering support.")

# -------------------------------
# VISUALIZATIONS
# -------------------------------
plot_top_items(encoded_df)
plot_item_distribution(encoded_df)
plot_support_confidence(rules)
plot_lift(rules)
plot_heatmap(rules)

# -------------------------------
# SAVE OUTPUT FILES
# -------------------------------
print("\n💾 Saving results...")

rules.to_csv("association_rules.csv", index=False)
encoded_df.sum().to_csv("item_frequencies.csv")

print("✅ Files saved successfully!")
print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")