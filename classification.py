import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class Classification:
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
        transactions['description'] = transactions['description'].str.replace('\bNAN\b', 'UNKNOWN', regex=True)

        description = transactions.loc[:, ['stock_code', 'description']]
        description = description.drop_duplicates(subset=['stock_code'])
        description = description[~description['stock_code'].str.contains(r'POST|\bD\b|\bS\b|\bM\b|C2|AMAZONFEE|BANK CHARGES|DOT|^GIFT_', regex=True)]

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

    def knn_classification(self):
        transactions = self.transactions
        dataset = pd.read_excel('dataset/dataset.xlsx')

        pred = transactions.iloc[:, 2:5].values

        x = dataset.iloc[:, 2:5].values
        y = dataset.iloc[:, -1].values

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.20, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_pred = sc.transform(pred)

        best_k = 11
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_pred)

        return transactions, y_pred