import numpy as np
from scipy.stats import t

class LinearRegression():
    def __init__(self, X, y, add_constant=True):
        if add_constant:
            self.X  = np.column_stack((np.ones(len(X)), X))
        elif X.ndim == 1:
            self.X = np.array(X).reshape(-1, 1)
        else:
            self.X = X
        self.y = y
        self.coeff = None

    def fit(self, model='OLS'):
        if model == 'OLS':
            self.coeff = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
            self.residuals = self.y - self.X @ self.coeff
            self.regression_se = np.sqrt(np.sum(self.residuals**2) / (len(self.y) - len(self.X.T)))
        else:
            #TODO: other models
            raise ValueError('Model not implemented')
        return self
    
    def tstatistics(self, r=0, estimator='default'):
        if estimator == 'default':
            self.coeff_est = estimator
            self.coeff_se = np.sqrt((self.regression_se ** 2) * np.diag(np.linalg.inv(self.X.T @ self.X)))
            t_stat = (self.coeff - r) / self.coeff_se
            return t_stat
        elif estimator == 'robust':
            #TODO : robust estiamtor
            raise ValueError('Model not implemented')
    
    def AIC(self):
        log_likelihood = (-len(self.y) / 2) * (np.log(self.regression_se ** 2) + np.log(2 * np.pi)) - \
            (self.residuals @ self.residuals.T / (2 * (self.regression_se ** 2)))
        return 2 * len(self.X.T) - 2 * log_likelihood
    
    def BIC(self):
        log_likelihood = (-len(self.y) / 2) * (np.log(self.regression_se ** 2) + np.log(2 * np.pi)) - \
            (self.residuals @ self.residuals.T / (2 * (self.regression_se ** 2)))
        return len(self.X.T) * np.log(len(y)) - 2 * log_likelihood
    
    def __str__(self):
        #TODO str
        return self
    

class AugmentedDickeyFuller():
    def __init__(self, y, intercept=True, trend=False, maxlag=0, autolag='AIC'):
        self.y = y
        self.intercept = intercept
        self.trend = trend
        self.maxlag = maxlag
        self.autolag = autolag
        self.coeff = None

    def fit(self):
        dependent = self.y.diff()[1:]
        independent = self.y.shift(1)[1:]
        if self.intercept:
            independent = np.column_stack((independent, np.ones(len(independent))))
        if self.trend:
            independent = np.column_stack((independent, np.arange(1, len(independent) + 1)))
        if self.autolag == None:
            if self.maxlag != 0:
                lags = np.column_stack([dependent.shift(i) for i in range(1, self.maxlag+1)])
                independent = np.column_stack((independent, lags))
                independent = independent[self.maxlag:]
                dependent = dependent[self.maxlag:]
        elif self.autolag == 'AIC':
            if self.maxlag != 0:
                AICs = []
                independent_aic = independent.copy()
                dependent_aic = dependent.copy()
                regression = LinearRegression(independent_aic, dependent_aic, add_constant=False)
                regression.fit()
                AICs.append(regression.AIC())
                lags = np.column_stack([dependent.shift(i) for i in range(1, self.maxlag+1)])
                for i in range(1, self.maxlag + 1):
                    independent_aic = np.column_stack((independent, lags[:, :i]))[i:]
                    dependent_aic = dependent[i:]
                    regression = LinearRegression(independent_aic, dependent_aic, add_constant=False)
                    regression.fit()
                    AICs.append(regression.AIC())
                optimal_lag = np.argmin(AICs)
                independent = np.column_stack((independent, lags[:, :optimal_lag]))[optimal_lag:]
                dependent = dependent[optimal_lag:]
        elif self.autolag == 'BIC':
            if self.maxlag != 0:
                BICs = []
                independent_bic = independent.copy()
                dependent_bic = dependent.copy()
                regression = LinearRegression(independent_bic, dependent_bic, add_constant=False)
                regression.fit()
                BICs.append(regression.BIC())
                lags = np.column_stack([dependent.shift(i) for i in range(1, self.maxlag+1)])
                for i in range(1, self.maxlag + 1):
                    independent_bic = np.column_stack((independent, lags[:, :i]))[i:]
                    dependent_bic = dependent[i:]
                    regression = LinearRegression(independent_bic, dependent_bic, add_constant=False)
                    regression.fit()
                    BICs.append(regression.BIC())
                optimal_lag = np.argmin(BICs)
                independent = np.column_stack((independent, lags[:, :optimal_lag]))[optimal_lag:]
                dependent = dependent[optimal_lag:]
        else:
            print('Unknown autolag method')
        regression = LinearRegression(independent, dependent, add_constant=False)
        regression.fit()
        self.coeff = regression.tstatistics()[0]
        return self
    
    def __str__(self):
        #TODO str
        return

class EngleGranger():
    def __init__(self, y, x, intercept=True, trend=False, maxlag=0, autolag='AIC'):
        self.y = y
        self.x = x
        self.intercept = intercept
        self.trend = trend
        self.maxlag = maxlag
        self.autolag = autolag
        self.coeff = None
    
    def fit(self):
        regressioon = LinearRegression(self.x, self.y, add_constant=True)
        regressioon.fit()
        self.residuals = regressioon.residuals
        self.coeff = regressioon.coeff
        ADF = AugmentedDickeyFuller(self.residuals, intercept=self.intercept, trend=self.trend, maxlag=self.maxlag, autolag=self.autolag)
        ADF.fit()
        self.ADFstat = ADF.coeff
        ecm_x = np.column_stack((self.x.diff()[1:], self.residuals.shift(1)[1:]))
        ecm_y = self.y.diff()[1:]
        ecm_regression = LinearRegression(ecm_x, ecm_y, add_constant=False)
        ecm_regression.fit()
        self.ecm_coeff = ecm_regression.coeff[1]
        self.ecm_tstat = ecm_regression.tstatistics()[1]
        return self
    
    def __str__(self):
        s = 'Engle-Granger Cointegration Test\n' + \
            '--------------------------------\n' + \
            'Beta from naive regression: ' + str(self.coeff[1]) + '\n' + \
            'ADF Statistic for residuals: ' + str(self.ADFstat) + '\n' + \
            'ECM Coefficient: ' + str(self.ecm_coeff) + '\n' + \
            'ECM t-statistic: ' + str(self.ecm_tstat) + '\n'
        return s
    
def tableADF(nobs,p):
    #Courtesy of Dr Diamond

    ''' 
    INPUTS:   nobs = # of observations
              p = order of time polynomial in the null-hypothesis
                 p = -1, no deterministic part
                 p =  0, for constant term
                 p =  1, for constant plus time-trend
                 p >  1, for higher order polynomial
    
    RETURNS :return critical values for the Zt statistic used in adf()
    '''


    zt =[[ -2.63467,   -1.95254 ,  -1.62044 ,  0.910216 ,   1.30508  ,  2.08088],
  [-3.63993,   -2.94935,   -2.61560,  -0.369306, -0.0116304,   0.666745],
  [-4.20045 ,  -3.54490 ,  -3.21450 ,  -1.20773 , -0.896215 , -0.237604],
  [-4.65813  , -3.99463  , -3.66223  , -1.69214  , -1.39031  ,-0.819931],
  [-5.07175   ,-4.39197   ,-4.03090   ,-2.06503   ,-1.78329,   -1.21830],
  [-5.45384,   -4.73277,   -4.39304 ,  -2.40333,   -2.15433 ,  -1.62357],
  [-5.82090 ,  -5.13053 ,  -4.73415  , -2.66466 ,  -2.39868  , -1.88193],
  [-2.53279  , -1.94976  , -1.62656   ,0.915249  ,  1.31679,    2.11787],
  [-3.56634   ,-2.93701   ,-2.61518,  -0.439283, -0.0498821 ,  0.694244],
  [-4.08920,   -3.46145,   -3.17093 ,  -1.25839 , -0.919533  ,-0.298641],
  [-4.56873 ,  -3.89966 ,  -3.59161  , -1.72543  , -1.44513,  -0.894085],
  [-4.97062  , -4.33552  , -4.00795   ,-2.12519   ,-1.85785 ,  -1.30566],
  [-5.26901   ,-4.62509   ,-4.29928,   -2.42113,   -2.15002   ,-1.65832],
  [-5.54856,   -4.95553,   -4.63476 ,  -2.71763 ,  -2.46508,   -1.99450],
  [-2.60249 ,  -1.94232 ,  -1.59497  , 0.912961  ,  1.30709 ,   2.02375],
  [-3.43911  , -2.91515  , -2.58414,  -0.404598, -0.0481033  , 0.538450],
  [-4.00519   ,-3.46110   ,-3.15517 ,  -1.25332 , -0.958071,  -0.320677],
  [-4.46919,   -3.87624,   -3.58887  , -1.70354  , -1.44034 , -0.920625],
  [-4.84725 ,  -4.25239 ,  -3.95439   ,-2.11382   ,-1.85495  , -1.26406],
  [-5.15555  , -4.59557  , -4.30149,   -2.41271,   -2.19370   ,-1.70447],
  [-5.46544   ,-4.89343   ,-4.58188 ,  -2.74151  ,-2.49723,   -2.02390],
  [-2.58559,   -1.94477,   -1.62458  , 0.905676,    1.30371 ,   2.01881],
  [-3.46419 ,  -2.91242 ,  -2.58837,  -0.410558 ,-0.0141618  , 0.665034],
  [-4.00090  , -3.45423  , -3.16252 ,  -1.24040  ,-0.937658,  -0.304433],
  [-4.45303   ,-3.89216   ,-3.61209  , -1.74246   ,-1.48280 , -0.906047],
  [-4.79484,   -4.22115,   -3.92941   ,-2.11434,   -1.83632  , -1.30274],
  [-5.15005 ,  -4.58359 ,  -4.30336,   -2.44972 ,  -2.21312,   -1.68330],
  [-5.42757  , -4.88604  , -4.60358 ,  -2.74044  , -2.50205 ,  -2.04008],
  [-2.65229   ,-1.99090   ,-1.66577  , 0.875165   , 1.27068  ,  2.04414],
  [-3.49260,   -2.87595,   -2.56885,  -0.416310, -0.0488941,   0.611200],
  [-3.99417 ,  -3.42290 ,  -3.13981 ,  -1.25096 , -0.950916 , -0.310521],
  [-4.42462  , -3.85645  , -3.56568  , -1.73108  , -1.45873  ,-0.934604],
  [-4.72243   ,-4.22262   ,-3.94435,   -2.10660   ,-1.84233,   -1.26702],
  [-5.12654,   -4.55072,   -4.24765 ,  -2.43456,   -2.18887 ,  -1.73081],
  [-5.46995 ,  -4.87930 ,  -4.57608,   -2.71226 ,  -2.48367  , -2.00597],
  [-2.63492  , -1.96775  , -1.62969 ,  0.904516  ,  1.31371    ,2.03286],
  [-3.44558   ,-2.84182   ,-2.57313  ,-0.469204,  -0.128358,   0.553411],
  [-3.99140,   -3.41543,   -3.13588,   -1.23585 , -0.944500,  -0.311271],
  [-4.43404 ,  -3.84922 ,  -3.56413 ,  -1.73854  , -1.48585 , -0.896978],
  [-4.75946  , -4.19562  , -3.91052  , -2.09997   ,-1.86034  , -1.32987],
  [-5.14042   ,-4.56772   ,-4.25699   ,-2.43882,   -2.18922,   -1.67371],
  [-5.39389,   -4.85343,   -4.57927,   -2.73497 ,  -2.49921 ,  -2.00247],
  [-2.58970 ,  -1.95674 ,  -1.61786 ,  0.902516  ,  1.32215  ,  2.05383],
  [-3.44036  , -2.86974  , -2.58294  ,-0.451590, -0.0789340   ,0.631864],
  [-3.95420   ,-3.43052   ,-3.13924,   -1.23328 , -0.938986,  -0.375491],
  [-4.40180,   -3.79982,   -3.52726 ,  -1.71598  , -1.44584 , -0.885303],
  [-4.77897 ,  -4.21672 ,  -3.93324  , -2.12309   ,-1.88431  , -1.33916],
  [-5.13508  , -4.56464  , -4.27617   ,-2.44358,   -2.18826   ,-1.72784],
  [-5.35071   ,-4.82097   ,-4.54914,   -2.73377 ,  -2.48874,   -2.01437],
  [-2.60653,   -1.96391,   -1.63477 ,  0.890881  ,  1.29296 ,   1.97163],
  [-3.42692 ,  -2.86280  , -2.57220  ,-0.463397, -0.0922419  , 0.613101],
  [-3.99299  , -3.41999 ,  -3.13524   ,-1.23857 , -0.929915  ,-0.337193],
  [-4.41297   ,-3.83582   ,-3.55450,   -1.72408  , -1.44915,  -0.872755],
  [-4.75811,   -4.18759,   -3.92599 ,  -2.12799   ,-1.88463 ,  -1.37118],
  [-5.08726 ,  -4.53617 ,  -4.26643  , -2.44694,   -2.19109  , -1.72329],
  [-5.33780  , -4.82542  , -4.54802   ,-2.73460 ,  -2.50726   ,-2.02927],
  [-2.58687   ,-1.93939   ,-1.63192,   0.871242  ,  1.26611,    1.96641],
  [-3.38577   ,-2.86443,   -2.57318 , -0.391939, -0.0498984 ,  0.659539],
  [-3.93785,   -3.39130 ,  -3.10317  , -1.24836 , -0.956349  ,-0.334478],
  [-4.39967 ,  -3.85724  , -3.55951   ,-1.74578  , -1.46374,  -0.870275],
  [-4.74764  , -4.20488   ,-3.91350,   -2.12384,   -1.88202 ,  -1.36853],
  [-5.07739   ,-4.52487,   -4.25185 ,  -2.43674 ,  -2.22289  , -1.72955],
  [-5.36172,   -4.81947 ,  -4.53837  , -2.74448  , -2.51367   ,-2.03065],
  [-2.58364 ,  -1.95730  , -1.63110   ,0.903082,    1.28613,    2.00605],
  [-3.45830  , -2.87104   ,-2.59369,  -0.451613 , -0.106025 ,  0.536687],
  [-3.99783   ,-3.43182,   -3.16171 ,  -1.26032  ,-0.956327  ,-0.305719],
  [-4.40298,   -3.86066 ,  -3.56940   ,-1.74588,   -1.48429,  -0.914111],
  [-4.84459 ,  -4.23012  , -3.93845   ,-2.15135 ,  -1.89876 ,  -1.39654],
  [-5.10571  , -4.56846   ,-4.28913,   -2.47637  , -2.22517  , -1.79586],
  [-5.39872   ,-4.86396,   -4.58525 ,  -2.78971   ,-2.56181   ,-2.14042]]
  
    i = round(nobs/50) +1
  
    if nobs < 50:
        i = i - 1 
   
    if i > 10:
        i = 10
 
    if p > 5:
        crit = [0,0,0,0,0,0]
  
    i = (i-1)*7 + p + 2  
    crit = zt[i-1]
    return crit