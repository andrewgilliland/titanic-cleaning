# titanic-cleaning

Example code to accompany the article [Cleaning and Exploring Data with Python](https://andrewgilliland.dev/articles/cleaning-and-exploring-data-with-python/).

Covers the full data cleaning and EDA workflow using the Titanic dataset:

1. Loading data with seaborn / pandas
2. Inspecting shape, types, and null counts
3. Handling missing values (drop, fill, indicator columns)
4. Fixing data types (categories, booleans, dates)
5. Removing duplicates
6. Exploratory data analysis (`describe`, `value_counts`, `groupby`, `corr`)
7. Visualizing with Matplotlib

## Running

```bash
uv run main.py
```

Charts are saved to the `output/` directory.

## Dependencies

- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
