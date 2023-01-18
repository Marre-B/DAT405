import pandas as pd
import matplotlib.pyplot as plt

# Load data
life_exp = pd.read_csv('Lab_1/life_expectancy.csv')
gdppc = pd.read_csv('Lab_1/gdp_per_capita.csv')

# Sort for 2018
life_exp_2018 = life_exp[life_exp["Year"] == 2018]
gdppc_2018 = gdppc[gdppc["Year"] == 2018]

# Merge and sort data
data_2018 = pd.merge(life_exp_2018, gdppc_2018, how="left", on="Entity")
data_2018 = data_2018[data_2018["Year_y"] == 2018]
data_2018 = data_2018.drop(columns=["Year_x", "Year_y", "417485-annotations", "Code_x", "Code_y"])
data_2018 = data_2018.rename(columns = {"Life expectancy - Sex: all - Age: at birth - Variant: estimates" : "Life expectancy"})

# Plot data
data_2018.plot.scatter(x="Life expectancy", y="GDP per capita", title="Life expectancy vs GDP per capita in 2018")

plt.xlabel("Life expectancy [Years]")
plt.ylabel("GDP per capita [USD]")
plt.show()

# B
print("Statistics for 2018:")
print(data_2018.describe())
print("-"*50)

mean_le = 72.631928
std_le = 7.772033

countries_le_std = data_2018[data_2018["Life expectancy"] > mean_le + std_le]
print(countries_le_std)

# C
mean_gdp = 19030.645795
std_gdp = 20286.783469

countries_highLE = data_2018[data_2018["Life expectancy"] > mean_le ]
countries_highLE_lowGDP = countries_highLE[countries_highLE["GDP per capita"] < mean_gdp]
print("Countries with high life expectancy and low GDP per capita:")
print(countries_highLE_lowGDP)

# D
countries_highGDP = data_2018[data_2018["GDP per capita"] > mean_gdp]
countries_highGDP_lowLE = countries_highGDP[countries_highGDP["Life expectancy"] < mean_le]
print("Countries with high GDP per capita and low life expectancy:")
print(countries_highGDP_lowLE)

# E
countries_lowLE = data_2018[data_2018["Life expectancy"] < mean_le]
print("Countries with low life expectancy:")
print(countries_lowLE["Entity"] == "India")