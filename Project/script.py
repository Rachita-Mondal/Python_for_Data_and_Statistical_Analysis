### Libraries
import pandas as pd
import numpy as np
from statistics import variance, stdev, mean, multimode, median
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
from numpy import linalg as la
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
### User Options

MENU = '''
(3) Module 3
(4) Module 4
(5) Module 5
(6) Module 6
(7) Module 7
(9) End Program
'''

MODULE_3 = '''
(1) Summary
(2) Plots
'''
SUMMARY = '''
(1) Mean
(2) Median
(3) Variance
(4) Standard Deviation
'''

PLOT = '''
(1) Histogram
(2) Scatterplot
'''

MODULE_4 = '''
(1) Test and ANOVA
(2) Solve equation
'''

TEST_ANOVA = '''
(1) One-way ANOVA
(2) Two-way ANOVA
(3) T-Test
'''

MODULE_5 = '''
(1) Regression
(2) Assumption Check
'''

ASSUMPTION = '''
(1) Linearity and Homoscedasticity
(2) Independence
(3) Normality
'''

MODULE_6 = '''
(1) WLS
(2) PCA
'''


MODULE_7 = '''
(1) Time Series Analysis
(2) Non-parametric methods
'''

TSA = '''
(1) Plot data
(2) Plot ACF
(3) Plot PACF
(4) Fit ARIMA
'''

NPM = '''
(1) KDE
(2) Wilcoxon Rank-Sum Test
(3) Kruskal-Wallis Test
'''



def load_data():
    start = True
    while start:
        print("Please upload your data:")
        file = input("Data file (without .csv): ")

        try:
            df = pd.read_csv(file + ".csv")
            print("\nPreview of your data:")
            df = df.dropna()
            print(df.head())
            
            correct = input("\nDoes it look right? (Y/N): ")

            if correct.upper() == "Y":   # handles y or Y
                start = False
                return df
            else:
                print("Try again!\n")

        except FileNotFoundError:
            print("File not found. Please check the name.\n")

def input_vector(name="Vector"):
    n = int(input(f"Enter size of {name}: "))
    vals = list(map(float, input(f"Enter values (space-separated) for {name}: ").split()))
    return np.array(vals)

def input_matrix(name="Matrix"):
    r = int(input(f"Enter number of rows for {name}: "))
    c = int(input(f"Enter number of cols for {name}: "))
    
    data = []
    print(f"Enter values row-wise for {name} (space separated):")
    for i in range(r):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        data.append(row)
    
    return np.array(data)


def module_3(df):
    choice = int(input("What would you like to do: "))
    if choice ==1:
        print(SUMMARY)
        choice_1 = int(input("Which summary would you like to have: "))
        if choice_1 == 1:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Which column would you want to use: ")
            print(f"Mean: {mean(df[column])}")
        elif choice_1 == 2:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Which column would you want to use: ")
            print(f"Median: {median(df[column])}")
        elif choice_1 == 3:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Which column would you want to use: ")
            print(f"Variance: {variance(df[column])}")
        elif choice_1 == 4:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Which column would you want to use: ")
            print(f"Standard Deviation: {stdev(df[column])}")

    elif choice == 2:
        print(PLOT)
        choice_1 = int(input("Which plot do you want: "))
        if choice_1 == 1:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Which column would you want to use: ")
            sns.histplot(data = df, x = column, bins = 50)
            plt.show()
        if choice_1 == 2:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column1 = input("Enter your X variable column: ")
            column2 = input("Enter your Y variable column: ")
            hue_column = input("Enter the column you want to use for hue: ")
            sns.scatterplot(data=df, x=column1, y=column2, hue=hue_column, alpha = 1)
            plt.show()

def module_4(df):
    choice = int(input("What would you like to do: "))
    
    if choice == 1:
        print(TEST_ANOVA)
        choice_1 = int(input("Which function do you want: "))
        
        if choice_1 == 1:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            Y = input("Enter your response variable: ")
            X = input("Enter your predictor variable: ")
            
            formula = f"{Y} ~ C({X})"
            model = smf.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(anova_table)

        elif choice_1 == 2:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            Y = input("Enter your response variable: ")
            X1 = input("Enter your first predictor variable: ")
            X2 = input("Enter your second predictor variable: ")
            
            formula = f"{Y} ~ C({X1}) + C({X2}) + C({X1}):C({X2})"
            model = smf.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(anova_table)

        elif choice_1 == 3:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("What column do you want the T-test for: ")
            null = float(input("What is your null value: "))
            t_stat, p_val = stats.ttest_1samp(df[column], popmean=null)
            print("One-sample t-test p-value:", p_val)
            if p_val < 0.05:
                print("Reject H0.")
            else:
                print("Fail to reject null H0.")
            
            print("\n\nActual mean: ", np.mean(df[column]))

    if choice == 2:
        A = input_matrix("A")
        b = input_vector("b")
        try:
            x = np.linalg.solve(A, b)
            det_A = la.det(A)
            print("Solution x:", x)
            if np.linalg.det(A) != 0:
                print("Unique solution exists")
            else:
                print("Unique solution does not exist")
        except:
            print("Cannot solve (singular matrix or mismatch)")

        
                
def module_5(df):
    choice = int(input("What would you like to do: "))
    if choice == 1:
        print("\nAvailable columns:\n")
        print(df.columns)
        print(df.info())
        Y = input("Enter your Y (response) variable: ")
        
        num_pred = input("Enter numerical X variables (comma-separated, or leave blank): ")
        num_pred = [v.strip() for v in num_pred.split(",") if v.strip() != ""]
        
        cat_pred = input("Enter categorical X variables (comma-separated, or leave blank): ")
        cat_pred = [f"C({v.strip()})" for v in cat_pred.split(",") if v.strip() != ""]
        
        predictors = num_pred + cat_pred
        
        formula = f"{Y} ~ {' + '.join(predictors)}"
        model = smf.ols(formula, data=df).fit()
        print(model.summary())
        

    elif choice == 2: 
        print("\nAvailable columns:\n")
        print(df.columns)
        print(df.info())
        Y = input("Enter your Y (response) variable: ")
        
        num_pred = input("Enter numerical X variables (comma-separated, or leave blank): ")
        num_pred = [v.strip() for v in num_pred.split(",") if v.strip() != ""]
        
        cat_pred = input("Enter categorical X variables (comma-separated, or leave blank): ")
        cat_pred = [f"C({v.strip()})" for v in cat_pred.split(",") if v.strip() != ""]
        
        predictors = num_pred + cat_pred
        
        formula = f"{Y} ~ {' + '.join(predictors)}"
        model = smf.ols(formula, data=df).fit()
        fitted = model.fittedvalues
        residuals = model.resid
        standardized_resid = (residuals - residuals.mean()) / residuals.std()
        print(ASSUMPTION)
        choice_1 = int(input("Which assumption would you like to check: "))
        if choice_1 == 1:
            print("The red smoothed line should be close to the dashed zero line if linearity holds.")
            plt.figure(figsize=(8, 5))
            sns.residplot(x=fitted, y=residuals, lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'linewidth': 1.5})
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.title('Residuals vs. Fitted')
            plt.xlabel('Fitted Values')
            plt.ylabel('Residuals')
            plt.tight_layout()
            plt.show()
        elif choice_1 == 2:
            print("Values between roughly 1.5 and 2.5 are generally considered acceptable.")
            dw = durbin_watson(residuals)
            print(f"Durbin-Watson statistic: {dw:.4f}")
        elif choice_1 == 3:
            print("If the residuals are normally distributed, the points should fall along the diagonal reference line.")
            fig, ax = plt.subplots(figsize=(6, 6))
            sm.qqplot(residuals, line='s', ax=ax, alpha=0.5)
            ax.set_title('Q-Q Plot of Residuals')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            plt.tight_layout()
            plt.show()



def module_6(df):
    choice = int(input("What would you like to do: "))
    
    if choice == 1:
        print("\nAvailable columns:\n")
        print(df.columns)
        print(df.info())
        Y = input("Enter your Y (response) variable: ")
        
        num_pred = input("Enter numerical X variables (comma-separated, or leave blank): ")
        num_pred = [v.strip() for v in num_pred.split(",") if v.strip() != ""]
        
        cat_pred = input("Enter categorical X variables (comma-separated, or leave blank): ")
        cat_pred = [f"C({v.strip()})" for v in cat_pred.split(",") if v.strip() != ""]
        
        predictors = num_pred + cat_pred
        
        formula = f"{Y} ~ {' + '.join(predictors)}"
        model = smf.ols(formula, data=df).fit()
        fitted_values = model.fittedvalues
        weights = 1 / (fitted_values ** 2)
        
        wls = smf.wls(formula, data=df, weights=weights).fit()
        print(wls.summary())

    elif choice == 2:
        print("\nAvailable columns:\n")
        print(df.columns)
        print(df.info())
        numeric_vars = input("Enter numerical variables (comma-separated, or leave blank): ")
        numeric_vars = [v.strip() for v in numeric_vars.split(",") if v.strip() != ""]
        X = df[numeric_vars]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        k = int(input("How many principal components do you want: "))
        seed = int(input("What random state do you want: "))
        pca = PCA(n_components= k, random_state = seed )
        X_pca = pca.fit_transform(X_scaled)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Total variance explained:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

        

        
        
        



def module_7(df):
    choice = int(input("What would you like to do: "))
    if choice == 1:
        print("\nAvailable columns:\n")
        print(df.columns)
        print(df.info())
        time_col = input("Enter the name of your time variable: ")
        Y = input("Enter your response variable: ")
        df_ts = df[[time_col, Y]].copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col])
        df_ts.set_index(time_col, inplace=True)
                
        print(df_ts.head())
        print(TSA)
        choice_1 = int(input("What would you like to do: "))
        if choice_1 == 1:
            df_ts.plot(title="Time Series Plot")
            plt.show()

        elif choice_1 == 2:
            lag = int(input("Up to what lag do you want to plot the autocorrelation function: "))
            plot_acf(df_ts, lags=lag)  
            plt.title('Autocorrelation Function (ACF)')
            plt.show()

        elif choice_1 == 3:
            lag = int(input("Up to what lag do you want to plot the partial autocorrelation function: "))
            plot_pacf(df_ts, lags=lag)  
            plt.title('Partial Autocorrelation Function (PACF)')
            plt.show()
        elif choice_1 == 4:
            p = int(input("AR(p) - How many past observation would you like to use: "))
            d = int(input("I - What order of differencing would you like to use: "))
            q = int(input("MA(q) - How many past forecast errors would you like to use: "))

            model = ARIMA(df_ts, order=(p, d, q)).fit()
            print(model.summary())
            model.plot_diagnostics(figsize=(10, 6))
            plt.show()
            
    elif choice == 2:
        print(NPM)
        choice_1 = int(input("What would you like to do: "))
        if choice_1 == 1:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            column = input("Enter numerical column for KDE plot: ")
            bw = float(input("What is bandwidth would you like to use: "))
            plt.figure(figsize=(8,5))
            sns.kdeplot(data=df, x=column, fill=True, bw_adjust=bw)
            plt.title(f"KDE Plot of {column}")
            plt.show()

        if choice_1 == 2:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            numeric_col = input("\nEnter numerical variable to compare: ")
            group_col = input("Enter grouping variable with two groups: ")
            groups = df[group_col].unique()
            group1 = df[df[group_col] == groups[0]][numeric_col]
            group2 = df[df[group_col] == groups[1]][numeric_col]
            stat, p = stats.mannwhitneyu(group1, group2)
            print(f"\nMann-Whitney U Statistic:, {stat:.4f}")
            print(f"p-value:, {p:.4f}")
            if p < 0.05:
                print("Reject H0.")
            else:
                print("Fail to reject null H0.")

        

        if choice_1 == 3:
            print("\nAvailable columns:\n")
            print(df.columns)
            print(df.info())
            numeric_col = input("\nEnter numerical variable to compare: ")
            group_col = input("Enter grouping variable with two or more groups: ")
            groups = df[group_col].unique()
            data_groups = []
            for g in groups:
                vals = df[df[group_col] == g][numeric_col]
                data_groups.append(vals)
            stat_kw, p_kw = stats.kruskal(*groups)
            print(f'\nKruskal-Wallis Test')
            print(f'  H-statistic: {stat_kw:.4f}')
            print(f'  p-value:     {p_kw:.4f}')
            if p_kw < 0.05:
                print('  REJECT H0 — at least one group differs significantly.')
                print('  --> Run pairwise tests to identify which pairs differ.')
            else:
                print('  FAIL TO REJECT H0.')
            
                    
        
            

def main():
    while True:
        print(MENU)
        choice = int(input("Select an option: "))

        # Exit
        if choice == 9:
            print("Exiting program. Goodbye!")
            break

        
        elif choice == 3:
            print("These are the available options: \n")
            print(MODULE_3)
            reload = input("Load new data? (Y/N): ")
            if reload == "Y":
                df = load_data()
            print(MODULE_3)

            module_3(df)

        elif choice == 4:
            print("These are the available options: \n")
            print(MODULE_4)
    
            reload = input("Load new data? (Y/N): ")
            if reload == "Y":
                df = load_data()
            print(MODULE_4)

            module_4(df)
            

        elif choice == 5:
            print("These are the available options: \n")
            print(MODULE_5)
            reload = input("Load new data? (Y/N): ")
            if reload == "Y":
                df = load_data()
            print(MODULE_5)

            module_5(df)

        elif choice == 6:
            print("These are the available options: \n")
            print(MODULE_6)
            reload = input("Load new data? (Y/N): ")
            if reload == "Y":
                df = load_data()
            print(MODULE_6)

            module_6(df)

        elif choice == 7:
            print("These are the available options: \n")
            print(MODULE_7)
            reload = input("Load new data? (Y/N): ")
            if reload == "Y":
                df = load_data()
            print(MODULE_7)
            

            module_7(df)

        

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
                        
        
            
    
            
            
        
            
            
    

        
            
            
        









































