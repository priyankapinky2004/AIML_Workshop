from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# 🛒 Sample transactions
transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'diaper', 'beer', 'bread'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['milk', 'bread', 'diaper'],
    ['milk', 'bread', 'eggs'],
    ['milk', 'diaper', 'cola'],
]

# 🔄 Encode the transaction data into a DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("🛒 Transaction DataFrame:")
print(df.head())

# 📊 Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("\n📊 Frequent Itemsets:")
print(frequent_itemsets)

# 📈 Generate association rules
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.4)
print("\n📈 Association Rules:")
print(rules)

# 🔽 Sort rules by support
rules_sorted = rules.sort_values(by='support', ascending=False)

# ✅ Drop duplicate rules based on antecedents and consequents
rules_sorted = rules_sorted.drop_duplicates(subset=['antecedents', 'consequents']).reset_index(drop=True)

# 🎯 Select top 20 rules
top20 = rules_sorted.head(20)

# 🖨️ Display key columns of top rules
print("\n📊 Top 20 Association Rules:")
print(top20[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
