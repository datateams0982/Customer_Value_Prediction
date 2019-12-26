# Literature Review

## Feature:
- Embedding of transaction data outperforms handcrafted sociodemographic variables on different topics: customer segmentation
- Engagement data: Login record, interaction?

## Imbalance data:
- SMOTE: Suitable for behavioral data

## Model
- Different frequency different model?
- 2 stage? first predict who will be active, than predict the LTV
- efficient daily update?
- seasonal effect adjustment? (Market trend)

--> Specifically financial market?

An Engagement-Based Customer Lifetime Value System for E-commerce

# Summarize:

## Problem definition
- First to define: The time period of customer value wanted to predict
- Rolling time window updating every day?
- time window looking back
- Target users: Separate new users without transaction record and those with, how about frequency? who are the ones interested?

## Data:
- Transaction Detail (Dynamic): Spot, future, sub
- Profit and Loss (Dynamic)
- Demographic variables: Age, gender, address(通訊OR戶籍), income(x)(replace by交易限額?), occupation(x), nationality(x)? Mortgage?(NOT MUCH)
- Engagement data: How about those not using online service ?(x)

## Data exploration:
- Check demographic variables and LTV
- Try different transformation of address data: location, city, house value, channel?
- Aggregated transaction/engagement information: frequency, others?
- total profit and loss
- previous total value and next year total value
- others?

## Feature Engineering:
- Demographic variables
- Usage of transaction/P&L/engagement data: Embedding (Stacked autoencoder? CNN? Data transformation?)

## System Architecture:

