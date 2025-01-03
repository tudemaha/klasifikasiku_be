import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


class ClassificationC45:
    def __init__(self, excel_file: bytes):
        self.transactions = pd.read_excel(BytesIO(excel_file))
        self.transactions = self.__preprocessing(self.transactions)

    def __preprocessing(self, transactions: pd.DataFrame) -> pd.DataFrame:
        transactions = transactions.astype({
            'stock_code': 'str',
            'description': 'str',
        })

        transactions['description'] = transactions['description'].str.upper()
        transactions['stock_code'] = transactions['stock_code'].str.upper()
        transactions['description'] = transactions['description'].str.replace(r'\bNAN\b', 'UNKNOWN', regex=True)

        description = transactions.loc[:, ['stock_code', 'description']]
        description = description.drop_duplicates(subset=['stock_code'])
        description = description[~description['stock_code'].str.contains(
            r'POST|\bD\b|\bS\b|\bM\b|C2|AMAZONFEE|BANK CHARGES|DOT|^GIFT_', regex=True
        )]

        month_quantity = transactions.loc[:, ['stock_code', 'quantity']].groupby('stock_code').sum().reset_index()
        max_quantity = transactions.loc[:, ['stock_code', 'quantity']].groupby('stock_code').max().reset_index()

        avg_price = transactions.loc[:, ['stock_code', 'price']].groupby('stock_code').mean().reset_index()
        avg_price = avg_price.rename(columns={'price': 'avg_price'})

        selling = transactions.loc[:, ['invoice_no', 'stock_code', 'quantity']]
        selling = selling.groupby(['invoice_no', 'stock_code']).count()
        selling = selling['quantity'].groupby('stock_code').count().to_frame()
        selling = selling.rename(columns={'quantity': 'selling'}).reset_index()

        quantity = month_quantity.copy()
        quantity.loc[(quantity['quantity'] < 0), 'quantity'] = quantity['quantity']
        quantity.loc[(quantity['quantity'] < 30) & (max_quantity['quantity'] > 10), 'quantity'] = 1

        clean = description.merge(quantity, on='stock_code', how='left')
        clean = clean.merge(selling, on='stock_code', how='left')
        clean = clean.merge(avg_price, on='stock_code', how='left').sort_values(by='stock_code')

        return clean

    def c4_5_classification(self):
        transactions = self.transactions
        dataset = pd.read_excel('dataset/dataset.xlsx')

        pred = transactions.iloc[:, 2:5].values

        x = dataset.iloc[:, 2:5].values
        y = dataset.iloc[:, -1].values

        # Split the dataset into training, validation, and testing sets
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.20, random_state=0)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.20, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_val = sc.transform(x_val)
        x_test = sc.transform(x_test)
        transactions_x = sc.transform(pred)

        # Build the decision tree
        decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        decision_tree.fit(x_train, y_train)

        # Post-pruning using validation data
        self.post_pruning(decision_tree, x_val, y_val)

        # Make predictions
        transactions_y = decision_tree.predict(transactions_x)

        y_pred = decision_tree.predict(x_test)
        f1 = f1_score(y_test, y_pred, pos_label='Laku')
        recall = recall_score(y_test, y_pred, pos_label='Laku')
        precision = precision_score(y_test, y_pred, pos_label='Laku')

        cm = confusion_matrix(y_test, y_pred, labels=['Laku', 'Tidak Laku'])
        print("Confusion Matrix:")
        print(cm)

        # Menghitung total data
        total_data = cm.sum()  # Menjumlahkan semua elemen dalam matriks
        print(f"Total data digunakan dalam Confusion Matrix: {total_data}")

        return transactions, transactions_y, f1, recall, precision

    def post_pruning(self, tree, x_val, y_val):
        """
        Perform post-pruning on a decision tree using validation data.
        """
        def is_leaf(node):
            return not hasattr(node, 'children_left') or tree.children_left[node] == -1

        def prune(node):
            if is_leaf(node):
                return

            # Prune left and right children first
            if tree.children_left[node] != -1:
                prune(tree.children_left[node])
            if tree.children_right[node] != -1:
                prune(tree.children_right[node])

            # Check error before and after pruning
            left_child = tree.children_left[node]
            right_child = tree.children_right[node]

            # Temporarily prune
            tree.children_left[node] = -1
            tree.children_right[node] = -1

            # Error after pruning
            pruned_pred = tree.predict(x_val)
            pruned_error = np.sum(pruned_pred != y_val)

            # Error before pruning
            tree.children_left[node] = left_child
            tree.children_right[node] = right_child
            full_pred = tree.predict(x_val)
            full_error = np.sum(full_pred != y_val)

            # Keep pruned if it has lower or equal error
            if pruned_error <= full_error:
                tree.children_left[node] = -1
                tree.children_right[node] = -1

        prune(0)  # Start pruning from the root node