---
description: 'follow these guidelines when evaluating data and plotting results'

---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

The plots should always be safed as pdf with name like f"plots/{name}/{variation}/{name}plot_description.pdf". 
Safe plots in two versions: one with a white background and one with a black background. The black version is for use in presentations, the white version is for use in papers. Consider also to adjust font colors for better visibility in each version. The white plot should not have a title. Display the plot with black background in the notebook, but save both versions. Use Latex for labels and titles in the plots, and make sure to use a consistent font size and style throughout the project. Always check that the plots are clear and easy to read, and that they accurately represent the underlying data. Consider using subplots to show multiple related plots together, and use appropriate spacing and layout to make them visually appealing. 

When evaluating data, always check the raw data and the intermediate steps of the analysis to make sure that the results are reasonable. If you find any issues, try to understand why they occur and how to fix them. When means are calculated, always check the distribution of the data to make sure that the mean is a reasonable summary statistic. If the data is skewed or has outliers, consider using a different summary statistic (e.g. median) or transforming the data before calculating the mean. Always report the sample size and any relevant statistics (e.g. standard deviation, confidence intervals) when presenting results. Include standard deviations as error bars in plots, and consider using boxplots or violin plots to show the distribution of the data. Always be transparent about any data processing steps (e.g. filtering, normalization) that were applied to the data, and provide code or documentation to allow others to reproduce the analysis. Print status updates after key steps or ones that take a long time, use e.g. tqdm for loops. 

When dealing with large datasets, consider using dask or other tools to handle the data efficiently. Always check the memory usage and performance of your code, and optimize it if necessary. Try to utilize dask's lazy evaluation features when working with large datasets and use as much parallelization as possible. When working with large datasets, consider using a subset of the data for testing and debugging before running the full analysis. Always document your code and analysis steps clearly, so that others can understand and reproduce your work.



When plotting results, make sure to label axes clearly and include units where appropriate. Use a consistent color scheme and style for all plots in the project. If you are comparing multiple conditions, use different colors or markers to distinguish them. Always include a legend if there are multiple conditions or groups in the plot. Make x and y axis limits appropriate to the data being plotted, and consider using log scales if the data spans several orders of magnitude. Finally, always check that the plots are clear and easy to read, and that they accurately represent the underlying data.

Implement new functions always so they line up with the existing code style and structure. If you are adding a new function, consider where it should be placed in the codebase and how it should be named. Follow the existing naming conventions and code organization to maintain consistency. Always write docstrings for new functions, explaining their purpose, inputs, outputs, and any relevant details. If you are modifying existing functions, make sure to update their docstrings accordingly. Always test your code thoroughly to ensure that it works as expected and does not introduce any bugs or issues.
Run lightweight cells after finishing coding, run other cells before if necessary.

Implement these guidelines in the analysis_unified.ipynb notebook and any other relevant notebooks or scripts in the project.
