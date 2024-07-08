import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency

# * FUNCTIONS
def create_daily_orders_df(df):
  daily_orders_df = df.resample(rule="D", on="order_estimated_delivery_date").agg({
    "order_id": "nunique",
    "price": "sum"
  }).reset_index()
  daily_orders_df.rename(columns={
      "order_estimated_delivery_date": "order_date",
      "order_id": "order_count",
      "price": "revenue"
  }, inplace=True)
  return daily_orders_df


def create_bystate_customer_df(df):
  bystate_customer_df = df.groupby(by="customer_state").customer_id.nunique().sort_values(ascending=False).reset_index()
  bystate_customer_df.rename(columns={
    "customer_id": "customer_count",
  }, inplace=True)
  return bystate_customer_df


def create_bycity_customer_df(df):
  bycity_customer_df = df.groupby(by="customer_city").order_id.nunique().sort_values(ascending=False).reset_index()
  bycity_customer_df.rename(columns={
    "order_id": "customer_count",
  }, inplace=True)
  return bycity_customer_df


def create_review_score_df(df):
  review_score_df = df.groupby(by="review_score").review_id.nunique().reset_index()
  review_score_df.rename(columns={
    "review_id": "review_count",
  }, inplace=True)
  return review_score_df


def create_byproduct_df(df):
  byproduct_df = df.groupby(by="product_category_name").product_id.nunique()
  return byproduct_df


def create_rfm_df(df):
  rfm_df = df.groupby("customer_id", as_index=False).agg({
    "order_estimated_delivery_date": "max",
    "order_id": "nunique",
    "price": "sum"
  })
  rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
  recent_date = rfm_df["max_order_timestamp"].max()
  rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
  rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
  
  return rfm_df


# * Build Main DataFrame
all_df = pd.read_csv("main_data.csv")
all_df.reset_index()

datetime_columns = [
    "order_purchase_timestamp",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column], errors='coerce', infer_datetime_format=True)

# * SIDEBAR
min_date = all_df["order_estimated_delivery_date"].min()
max_date = all_df["order_estimated_delivery_date"].max()

with st.sidebar:
  start_date, end_date = st.date_input(
    "Date Range", 
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)  
  )

main_df = all_df[
  (all_df["order_estimated_delivery_date"] >= str(start_date)) & (all_df["order_estimated_delivery_date"] <= str(end_date))
]

# * DATAFRAME
daily_orders_df = create_daily_orders_df(main_df)
bystate_customer_df = create_bystate_customer_df(main_df)
bycity_customer_df = create_bycity_customer_df(main_df)
byproduct_df = create_byproduct_df(main_df)
review_score_df = create_review_score_df(main_df)
rfm_df = create_rfm_df(main_df)

# * MAIN CONTENT
st.title("Dashboard")

## * DAILY ORDERS
st.header("Daily Orders")

fig, ax = plt.subplots(figsize=(10, 5))
col1, col2 = st.columns(2)

with col1:
  total_orders = daily_orders_df.order_count.sum()
  st.metric(label="Total Orders", value=total_orders)

with col2:
  total_revenue = daily_orders_df.revenue.sum()
  format_total_revenue = format_currency(total_revenue, 'BRL', locale='pt_BR')
  st.metric(label="Total Revenue", value=format_total_revenue)
  
sns.lineplot(
  data=daily_orders_df,
  x="order_date",
  y="order_count",
  markers="o",
)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

st.pyplot(fig)

## * CUSTOMER DEMOGRAPHICS
st.header("Customer Demographics")

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

sns.barplot(data=bystate_customer_df.head(8), x="customer_count", y="customer_state", ax=ax[0])
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_title("Customer Count by State")
ax[0].tick_params(axis="x", labelsize=10)
ax[0].tick_params(axis="y", labelsize=8)

sns.barplot(data=bycity_customer_df.head(8), x="customer_count", y="customer_city", ax=ax[1])
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_title("Customer Count by City")
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].tick_params(axis="x", labelsize=10)
ax[1].tick_params(axis="y", labelsize=8)
ax[1].invert_xaxis()

st.pyplot(fig)

## * REVIEW SCORE
st.header("Review Score")

fig, ax = plt.subplots()
col1, col2 = st.columns(2)

with col1:
  plt.pie(
    x=review_score_df.review_count, 
    labels=review_score_df.review_score, 
    autopct="%.1f%%", 
    textprops={'fontsize': 8}
  )

  st.pyplot(fig)

with col2:
  avg_review_df = round(main_df.review_score.mean(), 2)
  
  st.metric("Average Review Score", avg_review_df)

  fig, ax = plt.subplots()
  sns.barplot(
    data=review_score_df,
    x="review_score",
    y="review_count",
    color="green"
  )
  ax.set_xlabel("Rating Star", fontsize=15)
  ax.set_ylabel(None)
  st.pyplot(fig)
  
## * PRODUCT DEMOGRAPHICS
st.header("Product Demographics")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

best_product_df = byproduct_df.sort_values(ascending=False).reset_index().head(5)
worst_product_df = byproduct_df.sort_values(ascending=True).reset_index().head(5)

colors = ["green", "lightgrey", "lightgrey", "lightgrey", "lightgrey"]

sns.barplot(
  data=best_product_df,
  x="product_id",
  y="product_category_name",
  orient="h",
  palette=colors,
  ax=ax[0]
)
ax[0].set_title("Best Product", fontsize=15)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].tick_params(axis="x", labelsize=12)
ax[0].tick_params(axis="y", labelsize=12)

sns.barplot(
  data=worst_product_df,
  x="product_id",
  y="product_category_name",
  orient="h",
  palette=colors,
  ax=ax[1]
)
ax[1].set_title("Worst Product", fontsize=15)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].tick_params(axis="x", labelsize=12)
ax[1].tick_params(axis="y", labelsize=12)
ax[1].invert_xaxis()

st.pyplot(fig)