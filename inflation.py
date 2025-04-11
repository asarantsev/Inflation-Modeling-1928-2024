import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf

DF = pandas.read_excel('century.xlsx', sheet_name = 'data')
data = DF.values[:, 1:].astype(float)
volatility = data[1:, 0]
cpi = data[:, 3]
inflation = numpy.diff(numpy.log(cpi))
N = len(volatility) - 1

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')
    
print('Inflation = White Noise')
plots(inflation, 'inflation')
analysis(inflation, 'inflation')
print('Normalized Inflation = White Noise')
plots(inflation/volatility, 'inflation-vol')
analysis(inflation/volatility, 'inflation-vol')

print('Simple autoregression')
Reg = stats.linregress(inflation[:-1], numpy.diff(inflation))
print(Reg)
resid = numpy.array([numpy.diff(inflation)[k] - Reg.slope * inflation[k] - Reg.intercept for k in range(N)])
plots(resid, 'infl-simple')
analysis(resid, 'infl-simple')

print('Auto Regression with VIX')
RegDF = pandas.DataFrame({'Lag' : inflation[:-1]/volatility[1:], 'Volatility': 1, 'Constant' : 1/volatility[1:]})
Reg = sm.OLS(numpy.diff(inflation)/volatility[1:], RegDF).fit()
print(Reg.summary())
resid = Reg.resid
plots(resid, 'infl-vol')
analysis(resid, 'infl-vol')