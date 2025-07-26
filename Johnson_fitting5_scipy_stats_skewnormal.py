import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import glob

def read_data(folder):
    """
    This function reads the data from the folder containing the data sets.
    
    Folder: string containing the folder name
    """
    
    # Read all the .csv files in the folder
    files = glob.glob('*.csv')
    print(files)
    csv_list = []
    iso_code_list = []
    for file in files:
        df = pd.read_csv(file)
        # print(df.head())
        # remove the iso_code column, and make it a seperate list
        iso_code = df['iso_code'].tolist()
        df = df.drop('iso_code', axis = 1)
        csv_list.append(df)
        iso_code_list.append(iso_code)
    return csv_list, iso_code_list

df_list, iso_codes_list = read_data('Cluster_Data')
df_deaths = df_list[0]
iso_codes_deaths = iso_codes_list[0]


class SkewGaussianMixture:
    def __init__(self, n):
        self.n = n

    def _single_skew_gaussian(self, x, mu, sigma, skew, scale):
        return scale * stats.skewnorm.pdf(x, skew, mu, sigma)

    def skew_gaussian_mixture(self, x, *params):
        """
        This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
        So it consists of the sum of the Gaussian functions, with the parameters as arguments.
        
        Variables:
        x: variable
        params: list containing the parameters of the Gaussian functions
        """
        assert len(params) == 4 * self.n, "params should contain mu, sigma, skew, scale for each Gaussian"
        gaussians = [self._single_skew_gaussian(x, *params[i:i+4]) for i in range(0, len(params), 4)]
        return sum(gaussians)


function_N_list = [SkewGaussianMixture(n).skew_gaussian_mixture for n in range(2, 12)]   

# make an improved p0_list, where the mu's are equally spaced between 0 and the maximum value of the time_array
len_data = 1254
p0_list = []
for i in range(12):
    p0_list_i = []
    for j in range(i+1):
        p0_list_i.append(np.round(len_data/(i+2)*(j+1)))
        p0_list_i.append(50)
        p0_list_i.append(0)
        p0_list_i.append(2)
    p0_list.append(p0_list_i)
#remove first element of p0_list
p0_list = p0_list[1:]

skew_bound = 5
bound_list = [[[-100, 0, -skew_bound, 0] * n, [1354, 100, skew_bound, 1000] * n] for n in range(2, 12)]


# ranging from 1 to M, find the best N, based on the R^2 value between fitted curve and actual deaths data
def find_best_N(deaths_country, time_array, country_name):

    list_opt_params = []                                                                                        
    list_R2 = []
    list_moments = []
    list_maxima = []
    for i in range(len(function_N_list)):
        opt_params, cov_params = opt.curve_fit(function_N_list[i], time_array, deaths_country, 
                                               p0=p0_list[i],
                                               bounds=bound_list[i],
                                               absolute_sigma=False,
                                               method='trf',
                                            #    method='dogbox',
                                               ftol=1e-5
                                               )      
        list_opt_params.append(opt_params)

        fitted_curve = function_N_list[i](time_array, *opt_params)
        
        # calculate the number of local maxima in the fitted curve, no minimal distance between the maxima, and no minimal lentgh or height of the maxima, or distance to the edges or other maxima
        maxima = (np.diff(np.sign(np.diff(fitted_curve))) < 0).nonzero()[0] + 1
        list_maxima.append(maxima)
        # print('maxima = ', maxima)
        
        

        
        # plot the different gaussian functions
        moments_list = []
        for j in range(len(opt_params)//4):
            mu = opt_params[4*j]
            sigma = opt_params[4*j + 1]
            skew = opt_params[4*j + 2]
            scale = opt_params[4*j + 3]
            
            mean = stats.skewnorm.stats(skew, mu, sigma, moments='m')
            variance = stats.skewnorm.stats(skew, mu, sigma, moments='v')
            # std = stats.skewnorm.stats(skew, mu, sigma, moments='v')**0.5
            skewness = stats.skewnorm.stats(skew, mu, sigma, moments='s')
            
            moments = [mean, variance, skewness, scale]
            moments_list.append(moments)
            # # print all the moments:
            # print('j = ', j+1)
            # print('mu = ', mean))
            # print('SCALE = ', scale)
            # print('sigma = ', std)
            # print('skew = ', skewness)
            # print('kurto = ', stats.skewnorm.stats(skew, mu, sigma, moments='k'))
            # print('scale = ', scale)
            
            
            # gaussian = scale * stats.skewnorm.pdf(time_array, skew, mu, sigma)
            # plt.plot(time_array, gaussian, label='gaussian ' + str(j+1))
        
        R2 = 1 - np.sum((deaths_country - fitted_curve)**2) / np.sum((deaths_country - np.mean(deaths_country))**2)
        list_R2.append(R2)
        
        
        # # Plot the fitted curve and the actual data, also plot the different gaussian functions
        # plt.plot(time_array, deaths_country, label='deaths')
        # plt.plot(time_array, fitted_curve, label='fitted curve', alpha=1)        
        # plt.title(str(country_name) + ' fitted for N = ' + str(i+2) + 
        #           ' curves, with goodness of fit R2 = ' + str(R2.round(3)))
        # # plot axis meaning and labels
        # plt.xlabel('Time (days)')
        # plt.ylabel('Deaths (per day per million)')
        # plt.legend()
        # plt.show()
        
        for m in range(len(function_N_list)-i-1):
            moments_list.append([0, 0, 0, 0])
        # print('moments_list = ', moments_list)
        moments_array = np.array(moments_list)
        list_moments.append(moments_array)
  
    return list_R2, list_moments, list_maxima, list_opt_params


# MAYBE ADD THAT BASED ON THE R2 VALUE, THE BEST N IS CHOSEN, AND THEN THE PARAMETERS ARE OPTIMIZED AGAIN
# MAYBE ADD THAT THE PARAMETERS ARE OPTIMIZED AGAIN, BUT WITH A DIFFERENT METHOD
# TEST WITH DIFFERENT STARTING POINTS FOR THE PARAMETERS (AND MAYBE WITH DIFFERENT BOUNDS?) TO SHOW THE ROBUSTNESS

df_analysis = pd.DataFrame(columns=['Fitted_Parameters','Moments','R2','Local_Maxima'], index=iso_codes_deaths)
parameters_list = []
all_R2_list = []
# for k in range(df_deaths.shape[0]):
for k in range(len(iso_codes_deaths)):
# for k in range(138,141):
    # if k == 139:
    #     df_analysis.iloc[k,0] = 'Not enough data'
    #     df_analysis.iloc[k,1] = 'Not enough data'
    #     df_analysis.iloc[k,2] = 'Not enough data'
    #     df_analysis.iloc[k,3] = 'Not enough data'
    #     continue
        
    country_name = df_deaths.iloc[k,0]
    deaths_test_country = df_deaths.iloc[k,1:]
    
    if deaths_test_country.sum() == 0:
        df_analysis.iloc[k,0] = 'Not enough data'
        df_analysis.iloc[k,1] = 'Not enough data'
        df_analysis.iloc[k,2] = 'Not enough data'
        df_analysis.iloc[k,3] = 'Not enough data'
        continue
    
    # scale the data by dividing by the sum of the data
    # deaths_test_country = deaths_test_country / np.sum(deaths_test_country)
    # smooth the data by taking the moving average
    # deaths_test_country = deaths_test_country.rolling(window=10).mean()
    deaths_test_country = deaths_test_country.rolling(window=7).mean()
    # deaths_test_country = deaths_test_country.rolling(window=7).mean()
    # remove the NaN values
    deaths_test_country = deaths_test_country.dropna()
    time_test_array = np.arange(0, len(deaths_test_country))

    R2_list, moments_list, local_maxima, fitted_params = find_best_N(deaths_test_country, time_test_array, country_name)
    print('R2_list = ', R2_list)
    # get the index of the N where the R2 value does not improve more then 0.01 in the following two N's
    # make sure that the index calculation is not out of range
    # index = 0
    # for i in range(len(R2_list)):
    #     index = i
    #     if i == len(R2_list) - 1:
    #         break
    #     if R2_list[i+1] - R2_list[i] > 0.01:
    #         continue
    #     elif R2_list[i+1] - R2_list[i] < 0.01 and R2_list[i+2] - R2_list[i+2] < 0.01:
    #         break
    #     else:
    #         index = len(R2_list) - 1
    
    # # different heuristic: take the N where the R2 value is the highest, and then check if there is a smaller N where the R2 value is not more than 0.01 lower
    # index = np.argmax(R2_list)
    # for i in range(len(R2_list)):
    #     if R2_list[index] - R2_list[i] < 0.01:
    #         index = i
    #         break
    
    # # then do this again but with the value of 0.005 instead of 0.01, starting from the index found above
    # for i in range(index):
    #     if R2_list[index] - R2_list[i] < 0.005:
    #         index = i
    #         break
    
    
    
    # index = 0
    # for i in range(len(R2_list)):
    #     if i == len(R2_list) - 1:
            
        
    #     if R2_list[i+1] - R2_list[i] > 0.01:
    #         continue
    #     elif R2_list[i+1] - R2_list[i] > 0.01 and R2_list[i+2] - R2_list[i+2] > 0.01:
    #         index = i
    #         break
    #     else:
    #         index = len(R2_list) - 1
            
            
    # print('index = ', index)
    # get the best N
    # N = index + 2
    # print('N = ', N)
    # get the moment and scale parameters of the best N
    # moments_array = moments_list[index]
    # parameters_list.append(moments_array)
    all_R2_list.append(R2_list)
    # print('R2_list = ', R2_list)
    df_analysis.iloc[k,0] = fitted_params
    df_analysis.iloc[k,1] = moments_list
    df_analysis.iloc[k,2] = R2_list
    df_analysis.iloc[k,3] = local_maxima
    
    
# parameters_list = np.array(parameters_list)
# R2_list = find_best_N(deaths_test_country, time_test_array)
# all_R2_list = np.array(all_R2_list)


# save the dataframe to pickle in subfolder Fitted Curves
with open('Lognormal_Fitted_Curves_Deaths_N11.pkl', 'wb') as f:
    df_analysis.to_pickle(f)

# try to open the pickle file
df_try_out = pd.read_pickle('Lognormal_Fitted_Curves_Deaths.pkl')












# def skew_gaussian_function_2_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     return gaussian1 + gaussian2

# def skew_gaussian_function_3_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     return gaussian1 + gaussian2 + gaussian3

# def skew_gaussian_function_4_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4

# def skew_gaussian_function_5_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5

# def skew_gaussian_function_6_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6

# def skew_gaussian_function_7_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6, mu7, sigma7, skew7, scale7 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     gaussian7 = scale7 * stats.skewnorm.pdf(x, skew7, mu7, sigma7)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6 + gaussian7

# def skew_gaussian_function_8_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6, mu7, sigma7, skew7, scale7, mu8, sigma8, skew8, scale8 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     gaussian7 = scale7 * stats.skewnorm.pdf(x, skew7, mu7, sigma7)
#     gaussian8 = scale8 * stats.skewnorm.pdf(x, skew8, mu8, sigma8)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6 + gaussian7 + gaussian8

# def skew_gaussian_function_9_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6, mu7, sigma7, skew7, scale7, mu8, sigma8, skew8, scale8, mu9, sigma9, skew9, scale9 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     gaussian7 = scale7 * stats.skewnorm.pdf(x, skew7, mu7, sigma7)
#     gaussian8 = scale8 * stats.skewnorm.pdf(x, skew8, mu8, sigma8)
#     gaussian9 = scale9 * stats.skewnorm.pdf(x, skew9, mu9, sigma9)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6 + gaussian7 + gaussian8 + gaussian9

# def skew_gaussian_function_10_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the two Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6, mu7, sigma7, skew7, scale7, mu8, sigma8, skew8, scale8, mu9, sigma9, skew9, scale9, mu10, sigma10, skew10, scale10 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     gaussian7 = scale7 * stats.skewnorm.pdf(x, skew7, mu7, sigma7)
#     gaussian8 = scale8 * stats.skewnorm.pdf(x, skew8, mu8, sigma8)
#     gaussian9 = scale9 * stats.skewnorm.pdf(x, skew9, mu9, sigma9)
#     gaussian10 = scale10 * stats.skewnorm.pdf(x, skew10, mu10, sigma10)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6 + gaussian7 + gaussian8 + gaussian9 + gaussian10

# def skew_gaussian_function_11_mixture(x, *params):
#     """
#     This function should be the argument for 'f' in the scipy.optimize.curve_fit function.
#     So it consists of the sum of the three Gaussian functions, with the parameters as arguments.
    
#     Variables:
#     x: variable
#     params: list containing the parameters of the Gaussian functions
       
#     """    
#     mu1, sigma1, skew1, scale1, mu2, sigma2, skew2, scale2, mu3, sigma3, skew3, scale3, mu4, sigma4, skew4, scale4, mu5, sigma5, skew5, scale5, mu6, sigma6, skew6, scale6, mu7, sigma7, skew7, scale7, mu8, sigma8, skew8, scale8, mu9, sigma9, skew9, scale9, mu10, sigma10, skew10, scale10, mu11, sigma11, skew11, scale11 = params
#     gaussian1 = scale1 * stats.skewnorm.pdf(x, skew1, mu1, sigma1)
#     gaussian2 = scale2 * stats.skewnorm.pdf(x, skew2, mu2, sigma2)
#     gaussian3 = scale3 * stats.skewnorm.pdf(x, skew3, mu3, sigma3)
#     gaussian4 = scale4 * stats.skewnorm.pdf(x, skew4, mu4, sigma4)
#     gaussian5 = scale5 * stats.skewnorm.pdf(x, skew5, mu5, sigma5)
#     gaussian6 = scale6 * stats.skewnorm.pdf(x, skew6, mu6, sigma6)
#     gaussian7 = scale7 * stats.skewnorm.pdf(x, skew7, mu7, sigma7)
#     gaussian8 = scale8 * stats.skewnorm.pdf(x, skew8, mu8, sigma8)
#     gaussian9 = scale9 * stats.skewnorm.pdf(x, skew9, mu9, sigma9)
#     gaussian10 = scale10 * stats.skewnorm.pdf(x, skew10, mu10, sigma10)
#     gaussian11 = scale11 * stats.skewnorm.pdf(x, skew11, mu11, sigma11)
#     return gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian5 + gaussian6 + gaussian7 + gaussian8 + gaussian9 + gaussian10 + gaussian11

