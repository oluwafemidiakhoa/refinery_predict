import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:fFjyySqwkuBcRDZypgPNFgClOjCYjyfM@viaduct.proxy.rlwy.net:18945/railway"

# Connect to the database
engine = create_engine(DATABASE_URL)

# Load data
def load_data():
    query = "SELECT * FROM predictivemaintenance"
    data = pd.read_sql(query, engine)
    return data

data = load_data()
print(data.head())
