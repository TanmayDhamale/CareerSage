from surprise import SVD
from surprise import Dataset, Reader

# Load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'career_id', 'rating']], reader)

# Train model
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict career for a user
pred = algo.predict(user_id, career_id)